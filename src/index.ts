#!/usr/bin/env node

export interface LocalRunnerInfo {
  name: "tuned-tensor-local";
  status: "scaffold";
  description: string;
}

export function getLocalRunnerInfo(): LocalRunnerInfo {
  return {
    name: "tuned-tensor-local",
    status: "scaffold",
    description: "Local fine-tuning runner scaffold for single-GPU hosts.",
  };
}

function main(argv: string[]): void {
  const command = argv[2] ?? "info";

  if (command === "info" || command === "--help" || command === "-h") {
    const info = getLocalRunnerInfo();
    console.log(`${info.name}: ${info.description}`);
    console.log("Status: initial scaffold");
    return;
  }

  console.error(`Unknown command: ${command}`);
  process.exitCode = 1;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv);
}
