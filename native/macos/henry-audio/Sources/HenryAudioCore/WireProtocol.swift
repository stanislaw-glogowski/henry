import Foundation

public let wireProtocolVersion: UInt16 = 3

public enum WireMessageKind: UInt8 {
  case play = 1
  case stop = 2
  case shutdown = 3
  case duck = 4
  case restore = 5

  case ready = 128
  case capture = 129
  case playbackFinished = 130
  case stopped = 131
  case gainChanged = 132
  case diagnostic = 133
  case devices = 134
  case error = 255
}

public enum PlaybackStatus: UInt8 {
  case played = 0
  case interrupted = 1
}

/// Binary envelope: message kind, request identifier, payload length, payload.
public struct WireFrame: Equatable {
  public let kind: WireMessageKind
  public let requestID: UInt32
  public let payload: Data

  public init(kind: WireMessageKind, requestID: UInt32 = 0, payload: Data = Data()) {
    self.kind = kind
    self.requestID = requestID
    self.payload = payload
  }

  public func encode() -> Data {
    var data = Data()
    data.append(kind.rawValue)
    data.appendLittleEndian(requestID)
    data.appendLittleEndian(UInt32(payload.count))
    data.append(payload)
    return data
  }
}

public enum WireProtocolError: Error, Equatable, LocalizedError {
  case invalidMessageKind(UInt8)
  case invalidPayload(String)
  case unexpectedEndOfStream

  public var errorDescription: String? {
    switch self {
    case .invalidMessageKind(let value):
      "Invalid wire message kind: \(value)"
    case .invalidPayload(let message):
      "Invalid wire payload: \(message)"
    case .unexpectedEndOfStream:
      "Unexpected end of wire stream"
    }
  }
}

public final class WireFrameReader {
  private let handle: FileHandle

  public init(handle: FileHandle) {
    self.handle = handle
  }

  public func read() throws -> WireFrame? {
    guard let header = try readExactly(9, allowCleanEOF: true) else {
      return nil
    }

    let rawKind = header[header.startIndex]
    guard let kind = WireMessageKind(rawValue: rawKind) else {
      throw WireProtocolError.invalidMessageKind(rawKind)
    }

    let requestID = header.readLittleEndian(UInt32.self, at: 1)
    let payloadSize = header.readLittleEndian(UInt32.self, at: 5)
    let payload = try readExactly(Int(payloadSize), allowCleanEOF: false) ?? Data()
    return WireFrame(kind: kind, requestID: requestID, payload: payload)
  }

  private func readExactly(_ count: Int, allowCleanEOF: Bool) throws -> Data? {
    if count == 0 {
      return Data()
    }

    var data = Data()
    while data.count < count {
      guard let chunk = try handle.read(upToCount: count - data.count), !chunk.isEmpty else {
        if allowCleanEOF && data.isEmpty {
          return nil
        }
        throw WireProtocolError.unexpectedEndOfStream
      }
      data.append(chunk)
    }
    return data
  }
}

public final class WireFrameWriter: @unchecked Sendable {
  private let handle: FileHandle
  private let queue = DispatchQueue(label: "henry.audio.wire.output")

  public init(handle: FileHandle) {
    self.handle = handle
  }

  public func send(_ frame: WireFrame) {
    let encoded = frame.encode()
    queue.async { [handle] in
      do {
        try handle.write(contentsOf: encoded)
      } catch {
        // The peer owns process lifecycle; a closed pipe ends the helper.
      }
    }
  }

  public func flush() {
    queue.sync {}
  }
}

/// Interleaved Float32 audio carried by PLAY and CAPTURE messages.
public struct AudioPacket: Equatable {
  public let sampleRate: UInt32
  public let channels: UInt16
  public let samples: Data

  public init(sampleRate: UInt32, channels: UInt16, samples: Data) throws {
    guard sampleRate > 0 else {
      throw WireProtocolError.invalidPayload(
        "sample rate must be positive; got \(sampleRate)"
      )
    }
    guard channels > 0 else {
      throw WireProtocolError.invalidPayload(
        "channel count must be positive; got \(channels)"
      )
    }
    guard samples.count.isMultiple(of: MemoryLayout<Float>.size * Int(channels)) else {
      throw WireProtocolError.invalidPayload(
        "sample data is not frame-aligned: \(samples.count) bytes for \(channels) channels"
      )
    }
    self.sampleRate = sampleRate
    self.channels = channels
    self.samples = samples
  }

  public init(payload: Data) throws {
    guard payload.count >= 6 else {
      throw WireProtocolError.invalidPayload(
        "audio header is truncated: expected at least 6 bytes, got \(payload.count)"
      )
    }
    try self.init(
      sampleRate: payload.readLittleEndian(UInt32.self, at: 0),
      channels: payload.readLittleEndian(UInt16.self, at: 4),
      samples: payload.subdata(in: 6..<payload.count)
    )
  }

  public var frameCount: Int {
    samples.count / (MemoryLayout<Float>.size * Int(channels))
  }

  public func encode() -> Data {
    var data = Data()
    data.appendLittleEndian(sampleRate)
    data.appendLittleEndian(channels)
    data.append(samples)
    return data
  }
}

extension Data {
  mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
    var littleEndian = value.littleEndian
    Swift.withUnsafeBytes(of: &littleEndian) { append(contentsOf: $0) }
  }

  func readLittleEndian<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) -> T {
    let size = MemoryLayout<T>.size
    return subdata(in: offset..<(offset + size)).withUnsafeBytes {
      T(littleEndian: $0.loadUnaligned(as: T.self))
    }
  }
}
