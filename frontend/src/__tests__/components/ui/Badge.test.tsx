import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Badge from "@/components/ui/Badge";

describe("Badge", () => {
  it("renders text content", () => {
    render(<Badge>Active</Badge>);
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("applies default variant classes", () => {
    render(<Badge>Default</Badge>);
    expect(screen.getByText("Default").className).toContain("bg-gray-100");
  });

  it("applies success variant classes", () => {
    render(<Badge variant="success">OK</Badge>);
    expect(screen.getByText("OK").className).toContain("bg-green-100");
  });

  it("applies error variant classes", () => {
    render(<Badge variant="error">Fail</Badge>);
    expect(screen.getByText("Fail").className).toContain("bg-red-100");
  });

  it("applies warning variant classes", () => {
    render(<Badge variant="warning">Warn</Badge>);
    expect(screen.getByText("Warn").className).toContain("bg-yellow-100");
  });
});
