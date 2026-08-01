import Darwin
import Foundation
import HenryAudioCore

let reader = WireFrameReader(handle: .standardInput)
let writer = WireFrameWriter(handle: .standardOutput)
let voiceProcessingEnabled =
  ProcessInfo.processInfo.environment["HENRY_AUDIO_VOICE_PROCESSING"] != "0"
let controller = AudioEngineController(
  writer: writer,
  voiceProcessingEnabled: voiceProcessingEnabled
)

do {
  try controller.start()

  var readyPayload = Data()
  withUnsafeBytes(of: wireProtocolVersion.littleEndian) {
    readyPayload.append(contentsOf: $0)
  }
  writer.send(WireFrame(kind: .ready, payload: readyPayload))

  commandLoop: while let frame = try reader.read() {
    switch frame.kind {
    case .play:
      try controller.play(
        requestID: frame.requestID,
        packet: AudioPacket(payload: frame.payload)
      )
    case .stop:
      controller.stopPlayback()
      writer.send(WireFrame(kind: .stopped, requestID: frame.requestID))
    case .duck:
      controller.duckPlayback()
      writer.send(WireFrame(kind: .gainChanged, requestID: frame.requestID))
    case .restore:
      controller.restorePlayback()
      writer.send(WireFrame(kind: .gainChanged, requestID: frame.requestID))
    case .shutdown:
      break commandLoop
    default:
      throw WireProtocolError.invalidPayload(
        "unexpected command kind: \(frame.kind.rawValue)"
      )
    }
  }

  controller.shutdown()
  writer.flush()
} catch {
  writer.send(
    WireFrame(kind: .error, payload: Data(error.localizedDescription.utf8))
  )
  controller.shutdown()
  writer.flush()
  exit(EXIT_FAILURE)
}
