import CoreAudio
import Foundation

public struct AudioDeviceInfo: Codable, Equatable, Sendable {
  public let name: String
  public let identifier: String?

  public init(name: String, identifier: String? = nil) {
    self.name = name
    self.identifier = identifier
  }
}

public struct AudioDevicesPacket: Codable, Equatable, Sendable {
  public let input: AudioDeviceInfo
  public let output: AudioDeviceInfo

  public init(input: AudioDeviceInfo, output: AudioDeviceInfo) {
    self.input = input
    self.output = output
  }

  public func encode() throws -> Data {
    try JSONEncoder().encode(self)
  }
}

enum AudioDeviceResolver {
  static func defaults() -> AudioDevicesPacket {
    AudioDevicesPacket(
      input: defaultDevice(selector: kAudioHardwarePropertyDefaultInputDevice),
      output: defaultDevice(selector: kAudioHardwarePropertyDefaultOutputDevice)
    )
  }

  private static func defaultDevice(
    selector: AudioObjectPropertySelector
  ) -> AudioDeviceInfo {
    guard let deviceID = defaultDeviceID(selector: selector) else {
      return AudioDeviceInfo(name: "Unavailable")
    }

    return AudioDeviceInfo(
      name: stringProperty(
        deviceID: deviceID,
        selector: kAudioObjectPropertyName
      ) ?? "Unknown",
      identifier: stringProperty(
        deviceID: deviceID,
        selector: kAudioDevicePropertyDeviceUID
      )
    )
  }

  private static func stringProperty(
    deviceID: AudioDeviceID,
    selector: AudioObjectPropertySelector
  ) -> String? {
    var address = AudioObjectPropertyAddress(
      mSelector: selector,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let status = AudioObjectGetPropertyData(
      deviceID,
      &address,
      0,
      nil,
      &size,
      &value
    )
    guard status == noErr, let value else {
      return nil
    }
    return value.takeUnretainedValue() as String
  }

  private static func defaultDeviceID(
    selector: AudioObjectPropertySelector
  ) -> AudioDeviceID? {
    var address = AudioObjectPropertyAddress(
      mSelector: selector,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    var deviceID = AudioDeviceID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let status = AudioObjectGetPropertyData(
      AudioObjectID(kAudioObjectSystemObject),
      &address,
      0,
      nil,
      &size,
      &deviceID
    )
    guard status == noErr, deviceID != kAudioObjectUnknown else {
      return nil
    }
    return deviceID
  }
}
