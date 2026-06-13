//
//  main.swift
//  LLMBenchmark
//
//  Created by Mohamed Fawzi on 26/05/2026.
//

import CoreML
import Foundation
import Darwin

// MARK: - Memory helper

func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
    let kern = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kern == KERN_SUCCESS else { return -1 }
    return Double(info.phys_footprint) / 1_048_576
}

// MARK: - Config

let RUNS_PER_MODEL = 5

// MARK: - Model list
// isMistral = true  →  inputIds (camelCase) + causalMask + stateful KV-cache
// isMistral = false →  input_ids (snake_case) only

let base = URL(fileURLWithPath: "/Users/thisiscanaria/Developer/personal-projects/llm-edge-coreml")

let models: [(url: URL, name: String, isMistral: Bool)] = [
    (base.appendingPathComponent("models_fp16/phi4-mini-fp16.mlpackage"),  "Phi-4 Mini FP16", false),
    (base.appendingPathComponent("models_int8/phi4-mini-int8.mlpackage"),  "Phi-4 Mini INT8", false),
    (base.appendingPathComponent("models_int4/phi4-mini-int4.mlpackage"),  "Phi-4 Mini INT4", false),
    (base.appendingPathComponent("models_fp16/mistral-7b-fp16.mlpackage"), "Mistral 7B FP16", true),
    (base.appendingPathComponent("models_int8/mistral-7b-int8.mlpackage"), "Mistral 7B INT8", true),
    (base.appendingPathComponent("models_int4/mistral-7b-int4.mlpackage"), "Mistral 7B INT4", true),
]

// ← Change this index (0–5) for each run, then rebuild
let MODEL_INDEX = 5

// MARK: - Compile (cached)

let entry = models[MODEL_INDEX]

let cacheDir = base.appendingPathComponent("models_compiled")
try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
let safeName = entry.name.replacingOccurrences(of: " ", with: "_")
let cachedURL = cacheDir.appendingPathComponent("\(safeName).mlmodelc")

let compiledURL: URL
if FileManager.default.fileExists(atPath: cachedURL.path) {
    print("[\(entry.name)] Using cached compiled model…")
    compiledURL = cachedURL
} else {
    print("[\(entry.name)] Compiling .mlpackage → .mlmodelc")
    print("  (first time only — may take 5–30 min for large models)")
    do {
        let tempURL = try MLModel.compileModel(at: entry.url)
        try FileManager.default.moveItem(at: tempURL, to: cachedURL)
        compiledURL = cachedURL
        print("  Compilation done → \(cachedURL.lastPathComponent)")
    } catch {
        print("ERROR: Compilation failed — \(error)")
        exit(1)
    }
}

// MARK: - Load

let baselineMB = physFootprintMB()
print("\nBaseline memory : \(String(format: "%.0f MB", baselineMB))")
print("Loading \(entry.name)…")

let config = MLModelConfiguration()
config.computeUnits = .all

guard let model = try? MLModel(contentsOf: compiledURL, configuration: config) else {
    print("ERROR: Could not load \(entry.name)")
    exit(1)
}

let loadedMB = physFootprintMB()
print("✅  Loaded — memory after load : \(String(format: "%.0f MB (%.2f GB)", loadedMB, loadedMB / 1024))")

// MARK: - Build dummy input

func makeFeatures() throws -> MLFeatureProvider {
    if entry.isMistral {
        // Mistral 7B (Apple stateful model): inputIds + causalMask
        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[0] = NSNumber(value: Int32(1))
        // causalMask shape [batch=1, heads=1, q_len=1, kv_len=1], value 0 = attend
        let mask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float32)
        mask[0] = NSNumber(value: Float(0.0))
        return try MLDictionaryFeatureProvider(dictionary: ["inputIds": ids, "causalMask": mask])
    } else {
        // Phi-4 Mini: input_ids only
        let arr = try MLMultiArray(shape: [1, 1], dataType: .int32)
        arr[0] = NSNumber(value: Int32(1))
        return try MLDictionaryFeatureProvider(dictionary: ["input_ids": arr])
    }
}

guard let inputFeatures = try? makeFeatures() else {
    print("ERROR: Could not build input features")
    exit(1)
}

// MARK: - Inference helper

func runInference() throws {
    // Note: MLModel.newState() is iOS-only (explicitly unavailable on macOS).
    // For Mistral (stateful KV-cache model) we attempt stateless prediction as
    // a fallback — this may fail, which itself is a valid paper finding.
    let _ = try model.prediction(from: inputFeatures)
}

// MARK: - Warmup

print("\nWarmup pass…")
do {
    try runInference()
    print("  Warmup OK")
} catch {
    print("  Warmup FAILED: \(error)")
    print("\n⚠️  Inference not working — recording load-time memory only.")
    let peakMB = physFootprintMB()
    print("Peak memory (load only) : \(String(format: "%.0f MB  (%.2f GB)", peakMB, peakMB / 1024))")
    print("Press Enter to quit…")
    _ = readLine()
    exit(0)
}

// MARK: - Timed runs

print("Timed inference (\(RUNS_PER_MODEL) runs × 1 token)…")
var times: [Double] = []
for run in 1...RUNS_PER_MODEL {
    let start = Date()
    try? runInference()
    let elapsed = Date().timeIntervalSince(start)
    times.append(elapsed)
    print(String(format: "  Run %d: %.3f s  (%.1f tok/s)", run, elapsed, 1.0 / elapsed))
}

let medianTime = times.sorted()[RUNS_PER_MODEL / 2]
let peakMB = physFootprintMB()

print("""

╔══════════════════════════════════════════════════════╗
  \(entry.name)
  Peak memory (phys_footprint) : \(String(format: "%.0f MB  (%.2f GB)", peakMB, peakMB / 1024))
  Memory delta (load+infer)    : \(String(format: "%.0f MB  (%.2f GB)", peakMB - baselineMB, (peakMB - baselineMB) / 1024))
  Median decode latency        : \(String(format: "%.3f s / token", medianTime))
  Median throughput            : \(String(format: "%.2f tok/s", 1.0 / medianTime))
╚══════════════════════════════════════════════════════╝

>>> Record these numbers in your results table <<<
""")

print("Press Enter to quit…")
_ = readLine()
