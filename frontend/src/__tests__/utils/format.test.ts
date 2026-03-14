import { describe, it, expect } from "vitest";
import { formatBytes, formatMs, formatNumber } from "@/utils/format";

describe("formatBytes", () => {
  it("returns bytes for small values", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1023)).toBe("1023 B");
  });

  it("returns KB for values >= 1024", () => {
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("returns MB for values >= 1MB", () => {
    expect(formatBytes(1048576)).toBe("1.0 MB");
    expect(formatBytes(5242880)).toBe("5.0 MB");
  });
});

describe("formatMs", () => {
  it("returns ms for values < 1000", () => {
    expect(formatMs(0)).toBe("0ms");
    expect(formatMs(50)).toBe("50ms");
    expect(formatMs(999)).toBe("999ms");
  });

  it("returns seconds for values >= 1000", () => {
    expect(formatMs(1000)).toBe("1.00s");
    expect(formatMs(1500)).toBe("1.50s");
    expect(formatMs(12345)).toBe("12.35s");
  });
});

describe("formatNumber", () => {
  it("formats integers", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(42)).toBe("42");
  });

  it("formats large numbers with locale separators", () => {
    // Intl.NumberFormat uses locale-specific separators
    const result = formatNumber(1000000);
    expect(result).toContain("000");
  });
});
