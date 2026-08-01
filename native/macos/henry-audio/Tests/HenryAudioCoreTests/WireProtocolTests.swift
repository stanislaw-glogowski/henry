import Foundation
import Testing

@testable import HenryAudioCore

@Test func wireFrameEncodesHeaderAndPayload() throws {
  let frame = WireFrame(
    kind: .play,
    requestID: 42,
    payload: Data([1, 2, 3])
  )
  let encoded = frame.encode()

  #expect(encoded.count == 12)
  #expect(encoded[0] == WireMessageKind.play.rawValue)
  #expect(encoded.readLittleEndian(UInt32.self, at: 1) == 42)
  #expect(encoded.readLittleEndian(UInt32.self, at: 5) == 3)
  #expect(encoded.suffix(3) == Data([1, 2, 3]))
}

@Test func audioPacketRoundTripsAndValidatesAlignment() throws {
  let samples: [Float] = [0.25, -0.5, 0.75, -1]
  let packet = try AudioPacket(
    sampleRate: 16_000,
    channels: 2,
    samples: samples.withUnsafeBytes { Data($0) }
  )
  let decoded = try AudioPacket(payload: packet.encode())

  #expect(decoded == packet)
  #expect(decoded.frameCount == 2)

  #expect(throws: WireProtocolError.self) {
    try AudioPacket(sampleRate: 16_000, channels: 2, samples: Data([0]))
  }
}

@Test func audioDevicesPacketEncodesStructuredDeviceInformation() throws {
  let packet = AudioDevicesPacket(
    input: AudioDeviceInfo(name: "MacBook Microphone", identifier: "input-1"),
    output: AudioDeviceInfo(name: "Studio Display", identifier: "output-1")
  )

  let decoded = try JSONDecoder().decode(AudioDevicesPacket.self, from: packet.encode())

  #expect(decoded == packet)
}

@Test func errorsIdentifyTheirOwningLayer() {
  #expect(AudioEngineError.notStarted.errorDescription == "Audio engine is not started")
  #expect(
    WireProtocolError.invalidPayload("bad packet").errorDescription
      == "Invalid wire payload: bad packet"
  )
}
