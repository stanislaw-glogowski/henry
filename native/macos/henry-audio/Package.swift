// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "henry-audio",
  platforms: [
    .macOS(.v14)
  ],
  products: [
    .executable(name: "henry-audio", targets: ["HenryAudio"])
  ],
  targets: [
    .target(name: "HenryAudioCore"),
    .executableTarget(
      name: "HenryAudio",
      dependencies: ["HenryAudioCore"]
    ),
    .testTarget(
      name: "HenryAudioCoreTests",
      dependencies: ["HenryAudioCore"]
    ),
  ]
)
