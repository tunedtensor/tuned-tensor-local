import assert from "node:assert/strict";
import { test } from "node:test";
import { parseTrainingProgressLine } from "../src/process-training.js";

test("parses trainer metric dictionaries as progress snapshots", () => {
  const parsed = parseTrainingProgressLine("{'loss': '0.351', 'grad_norm': '0.6381', 'learning_rate': '4.098e-08', 'epoch': '1'}");

  assert.deepEqual(parsed, {
    loss: 0.351,
    grad_norm: 0.6381,
    learning_rate: 4.098e-8,
    epoch: 1,
    percent: 100,
  });
});

test("parses tqdm progress lines as progress snapshots", () => {
  const parsed = parseTrainingProgressLine(" 98%|█████████▊| 238/244 [29:25<00:48,  8.02s/it]");

  assert.deepEqual(parsed, {
    percent: 98,
    step: 238,
    total_steps: 244,
    elapsed: "29:25",
    eta: "00:48",
    rate: "8.02s/it",
  });
});
