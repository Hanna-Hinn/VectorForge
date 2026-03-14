import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Button from "@/components/ui/Button";

describe("Button", () => {
  it("renders children text", () => {
    render(<Button>Click me</Button>);
    expect(
      screen.getByRole("button", { name: "Click me" }),
    ).toBeInTheDocument();
  });

  it("is disabled when loading", () => {
    render(<Button loading>Save</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("shows loading text when loading", () => {
    render(<Button loading>Save</Button>);
    expect(screen.getByRole("button")).toHaveTextContent("Loading");
  });

  it("applies variant classes", () => {
    render(<Button variant="danger">Delete</Button>);
    const btn = screen.getByRole("button");
    expect(btn.className).toContain("bg-red-600");
  });

  it("applies size classes", () => {
    render(<Button size="lg">Big</Button>);
    const btn = screen.getByRole("button");
    expect(btn.className).toContain("py-2.5");
  });

  it("is disabled via disabled prop", () => {
    render(<Button disabled>Nope</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });
});
