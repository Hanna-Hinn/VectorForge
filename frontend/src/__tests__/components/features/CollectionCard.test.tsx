import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import CollectionCard from "@/components/features/CollectionCard";
import type { Collection } from "@/types/models";

const MOCK_COLLECTION: Collection = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "test-docs",
  description: "A test collection",
  embedding_config: null,
  chunking_config: null,
  created_at: "2025-01-01T00:00:00Z",
  updated_at: null,
};

function renderCard(onDelete = vi.fn()) {
  return render(
    <MemoryRouter>
      <CollectionCard collection={MOCK_COLLECTION} onDelete={onDelete} />
    </MemoryRouter>,
  );
}

describe("CollectionCard", () => {
  it("renders collection name", () => {
    renderCard();
    expect(screen.getByText("test-docs")).toBeInTheDocument();
  });

  it("renders collection description", () => {
    renderCard();
    expect(screen.getByText("A test collection")).toBeInTheDocument();
  });

  it("has navigation links", () => {
    renderCard();
    expect(screen.getByText("Documents")).toBeInTheDocument();
    expect(screen.getByText("Query")).toBeInTheDocument();
    expect(screen.getByText("Analytics")).toBeInTheDocument();
  });

  it("calls onDelete when delete button is clicked", async () => {
    const onDelete = vi.fn();
    renderCard(onDelete);
    await userEvent.click(screen.getByText("Delete"));
    expect(onDelete).toHaveBeenCalledWith(MOCK_COLLECTION.id);
  });
});
