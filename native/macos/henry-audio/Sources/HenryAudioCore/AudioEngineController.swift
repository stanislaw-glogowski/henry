@preconcurrency import AVFAudio
import Foundation

private let captureChannels: UInt16 = 1

public enum AudioEngineError: Error, LocalizedError {
  case notStarted
  case configuration(String)
  case conversion(String)

  public var errorDescription: String? {
    switch self {
    case .notStarted:
      "Audio engine is not started"
    case .configuration(let message):
      "Audio engine configuration failed: \(message)"
    case .conversion(let message):
      "Audio conversion failed: \(message)"
    }
  }
}

private final class AudioBufferBox: @unchecked Sendable {
  let buffer: AVAudioPCMBuffer

  init(_ buffer: AVAudioPCMBuffer) {
    self.buffer = buffer
  }
}

private final class AudioBufferSupplier: @unchecked Sendable {
  private var buffer: AVAudioPCMBuffer?
  private let exhaustedStatus: AVAudioConverterInputStatus

  init(
    buffer: AVAudioPCMBuffer,
    exhaustedStatus: AVAudioConverterInputStatus
  ) {
    self.buffer = buffer
    self.exhaustedStatus = exhaustedStatus
  }

  func next(status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
    guard let buffer else {
      status.pointee = exhaustedStatus
      return nil
    }
    self.buffer = nil
    status.pointee = .haveData
    return buffer
  }
}

/// Owns Henry's full-duplex AVAudioEngine and all device-side audio state.
public final class AudioEngineController: @unchecked Sendable {
  private let engine = AVAudioEngine()
  private let player = AVAudioPlayerNode()
  private let writer: WireFrameWriter
  private let voiceProcessingEnabled: Bool
  private let captureQueue = DispatchQueue(label: "henry.audio.capture")
  private let volumeQueue = DispatchQueue(label: "henry.audio.volume")
  private let playbackLock = NSLock()

  private var captureDiagnosticSent = false
  private var captureDiagnosticSampleCount = 0
  private var captureDiagnosticSquaredSum = 0.0
  private var captureDiagnosticPeak = Float.zero
  private var pendingPlayback: Set<UInt32> = []
  private var playbackFormat: AVAudioFormat?
  private var started = false

  public init(writer: WireFrameWriter, voiceProcessingEnabled: Bool = true) {
    self.writer = writer
    self.voiceProcessingEnabled = voiceProcessingEnabled
  }

  public func start() throws {
    guard !started else {
      return
    }

    let inputNode = engine.inputNode
    if voiceProcessingEnabled {
      // One voice-processing engine owns input and output, allowing macOS to use
      // Henry's playback as the echo-cancellation reference signal.
      try inputNode.setVoiceProcessingEnabled(true)
      inputNode.isVoiceProcessingInputMuted = false
      guard inputNode.isVoiceProcessingEnabled else {
        throw AudioEngineError.configuration(
          "voice processing did not become active"
        )
      }
    }

    guard
      let playbackFormat = AVAudioFormat(
        standardFormatWithSampleRate: 48_000,
        channels: 1
      )
    else {
      throw AudioEngineError.configuration(
        "cannot create the 48 kHz mono playback format"
      )
    }
    self.playbackFormat = playbackFormat

    engine.attach(player)
    engine.connect(player, to: engine.mainMixerNode, format: playbackFormat)

    inputNode.installTap(onBus: 0, bufferSize: 1_024, format: nil) {
      [weak self] buffer, _ in
      self?.receiveCapture(buffer)
    }

    engine.prepare()
    try engine.start()
    if voiceProcessingEnabled {
      // VPIO initialization may restore its input mute state while the engine
      // starts, so apply the full-duplex capture state after initialization.
      inputNode.isVoiceProcessingInputMuted = false
      inputNode.isVoiceProcessingBypassed = false
    }
    let devices = AudioDeviceResolver.defaults()
    writer.send(WireFrame(kind: .devices, payload: try devices.encode()))
    sendDiagnostic(
      "Voice processing CONFIGURED: enabled=\(inputNode.isVoiceProcessingEnabled), "
        + "input_muted=\(inputNode.isVoiceProcessingInputMuted), "
        + "bypassed=\(inputNode.isVoiceProcessingBypassed), "
        + "agc=\(inputNode.isVoiceProcessingAGCEnabled), "
        + "input_format=\(inputNode.outputFormat(forBus: 0)), "
        + "output_format=\(engine.outputNode.inputFormat(forBus: 0))"
    )
    started = true
  }

  public func play(requestID: UInt32, packet: AudioPacket) throws {
    guard started, let playbackFormat else {
      throw AudioEngineError.notStarted
    }

    let input = try makeBuffer(packet: packet)
    let output = try convert(input, to: playbackFormat)

    playbackLock.withLock {
      _ = pendingPlayback.insert(requestID)
    }

    player.scheduleBuffer(
      output,
      completionCallbackType: .dataPlayedBack
    ) { [weak self] _ in
      self?.finishPlayback(requestID: requestID)
    }

    if !player.isPlaying {
      player.play()
    }
  }

  public func stopPlayback() {
    let requestIDs = playbackLock.withLock {
      let values = pendingPlayback
      pendingPlayback.removeAll()
      return values
    }

    player.stop()
    for requestID in requestIDs {
      writer.send(
        WireFrame(
          kind: .playbackFinished,
          requestID: requestID,
          payload: Data([PlaybackStatus.interrupted.rawValue])
        )
      )
    }
  }

  public func duckPlayback() {
    rampPlaybackVolume(to: 0.18, duration: 0.08)
  }

  public func restorePlayback() {
    rampPlaybackVolume(to: 1, duration: 0.12)
  }

  public func shutdown() {
    guard started else {
      return
    }
    stopPlayback()
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    started = false
  }

  private func receiveCapture(_ buffer: AVAudioPCMBuffer) {
    sendCaptureDiagnostic(buffer)
    guard let copy = copyBuffer(buffer) else {
      return
    }
    let box = AudioBufferBox(copy)
    captureQueue.async { [weak self, box] in
      self?.processCapture(box.buffer)
    }
  }

  private func processCapture(_ input: AVAudioPCMBuffer) {
    do {
      guard let channel = input.floatChannelData?[0] else {
        throw AudioEngineError.conversion(
          "capture format \(input.format) does not provide Float32 channel data"
        )
      }
      let sampleRate = input.format.sampleRate
      guard sampleRate > 0, sampleRate <= Double(UInt32.max) else {
        throw AudioEngineError.configuration(
          "capture sample rate \(sampleRate) is invalid"
        )
      }
      let sampleData = Data(
        bytes: channel,
        count: Int(input.frameLength) * MemoryLayout<Float>.size
      )
      let packet = try AudioPacket(
        sampleRate: UInt32(sampleRate.rounded()),
        channels: captureChannels,
        samples: sampleData
      )
      writer.send(WireFrame(kind: .capture, payload: packet.encode()))
    } catch {
      sendError(error)
    }
  }

  private func finishPlayback(requestID: UInt32) {
    let wasPending = playbackLock.withLock {
      pendingPlayback.remove(requestID) != nil
    }
    guard wasPending else {
      return
    }
    writer.send(
      WireFrame(
        kind: .playbackFinished,
        requestID: requestID,
        payload: Data([PlaybackStatus.played.rawValue])
      )
    )
  }

  private func rampPlaybackVolume(to target: Float, duration: TimeInterval) {
    volumeQueue.async { [weak self] in
      guard let self else {
        return
      }
      let start = player.volume
      let steps = 8
      let stepDuration = duration / Double(steps)
      for step in 1...steps {
        let progress = Float(step) / Float(steps)
        player.volume = start + (target - start) * progress
        Thread.sleep(forTimeInterval: stepDuration)
      }
    }
  }

  private func makeBuffer(packet: AudioPacket) throws -> AVAudioPCMBuffer {
    guard
      let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(packet.sampleRate),
        channels: AVAudioChannelCount(packet.channels),
        interleaved: false
      ),
      let buffer = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(packet.frameCount)
      ), let channelData = buffer.floatChannelData
    else {
      throw AudioEngineError.configuration(
        "cannot allocate a playback buffer for \(packet.channels) channels at "
          + "\(packet.sampleRate) Hz"
      )
    }

    let samples = packet.samples.withUnsafeBytes {
      Array($0.bindMemory(to: Float.self))
    }
    let channelCount = Int(packet.channels)
    for channelIndex in 0..<channelCount {
      let destination = channelData[channelIndex]
      for frameIndex in 0..<packet.frameCount {
        destination[frameIndex] = samples[frameIndex * channelCount + channelIndex]
      }
    }
    buffer.frameLength = AVAudioFrameCount(packet.frameCount)
    return buffer
  }

  private func convert(
    _ input: AVAudioPCMBuffer,
    to outputFormat: AVAudioFormat
  ) throws -> AVAudioPCMBuffer {
    if input.format == outputFormat {
      return input
    }
    guard let converter = AVAudioConverter(from: input.format, to: outputFormat) else {
      throw AudioEngineError.conversion(
        "unsupported playback conversion from \(input.format) to \(outputFormat)"
      )
    }

    let ratio = outputFormat.sampleRate / input.format.sampleRate
    let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * ratio) + 16)
    guard
      let output = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity: capacity
      )
    else {
      throw AudioEngineError.conversion(
        "cannot allocate a converted playback buffer with capacity \(capacity)"
      )
    }

    let supplier = AudioBufferSupplier(
      buffer: input,
      exhaustedStatus: .endOfStream
    )
    var conversionError: NSError?
    let status = converter.convert(to: output, error: &conversionError) {
      _, inputStatus in
      supplier.next(status: inputStatus)
    }
    if let conversionError {
      throw conversionError
    }
    guard status != .error else {
      throw AudioEngineError.conversion(
        "playback converter returned status \(String(describing: status))"
      )
    }
    return output
  }

  private func copyBuffer(_ source: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
    guard
      let copy = AVAudioPCMBuffer(
        pcmFormat: source.format,
        frameCapacity: source.frameLength
      )
    else {
      return nil
    }
    copy.frameLength = source.frameLength

    let sourceBuffers = UnsafeMutableAudioBufferListPointer(
      UnsafeMutablePointer(mutating: source.audioBufferList)
    )
    let destinationBuffers = UnsafeMutableAudioBufferListPointer(
      copy.mutableAudioBufferList
    )
    guard sourceBuffers.count == destinationBuffers.count else {
      return nil
    }

    for index in sourceBuffers.indices {
      guard let sourceData = sourceBuffers[index].mData,
        let destinationData = destinationBuffers[index].mData
      else {
        return nil
      }
      let byteCount = Int(sourceBuffers[index].mDataByteSize)
      memcpy(destinationData, sourceData, byteCount)
      destinationBuffers[index].mDataByteSize = UInt32(byteCount)
    }
    return copy
  }

  private func sendError(_ error: Error) {
    writer.send(
      WireFrame(
        kind: .error,
        payload: Data(error.localizedDescription.utf8)
      )
    )
  }

  private func sendCaptureDiagnostic(_ buffer: AVAudioPCMBuffer) {
    guard !captureDiagnosticSent else {
      return
    }

    guard let channels = buffer.floatChannelData else {
      captureDiagnosticSent = true
      sendDiagnostic(
        "raw_capture format=\(buffer.format), frames=\(buffer.frameLength), "
          + "float_samples=unavailable"
      )
      return
    }

    let frameCount = Int(buffer.frameLength)
    let channelCount = Int(buffer.format.channelCount)
    for channelIndex in 0..<channelCount {
      let channel = channels[channelIndex]
      for frameIndex in 0..<frameCount {
        let sample = channel[frameIndex]
        captureDiagnosticPeak = max(captureDiagnosticPeak, abs(sample))
        captureDiagnosticSquaredSum += Double(sample * sample)
      }
    }
    captureDiagnosticSampleCount += frameCount * channelCount
    guard captureDiagnosticSampleCount >= 4_800 else {
      return
    }

    captureDiagnosticSent = true
    let rms = sqrt(
      captureDiagnosticSquaredSum / Double(captureDiagnosticSampleCount)
    )
    sendDiagnostic(
      "raw_capture format=\(buffer.format), frames=\(buffer.frameLength), "
        + "rms=\(rms), peak=\(captureDiagnosticPeak)"
    )
  }

  private func sendDiagnostic(_ message: String) {
    writer.send(
      WireFrame(
        kind: .diagnostic,
        payload: Data(message.utf8)
      )
    )
  }
}
