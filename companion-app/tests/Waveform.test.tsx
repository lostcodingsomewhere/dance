import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Waveform } from "../src/components/Waveform";

describe("Waveform", () => {
  it("renders one rect per peak", () => {
    render(<Waveform peaks={[0.1, 0.5, 0.9, 0.2]} />);
    const bars = screen.getAllByTestId("waveform-bar");
    expect(bars.length).toBe(4);
  });

  it("omits playhead when position is undefined", () => {
    render(<Waveform peaks={[0.1, 0.5, 0.9, 0.2]} />);
    expect(screen.queryByTestId("waveform-playhead")).toBeNull();
  });

  it("renders playhead at correct x when position set", () => {
    render(<Waveform peaks={[0.1, 0.5, 0.9, 0.2]} position={0.5} />);
    const playhead = screen.getByTestId("waveform-playhead");
    expect(playhead.getAttribute("x1")).toBe("50");
  });

  it("renders placeholder for empty peaks", () => {
    render(<Waveform peaks={[]} />);
    expect(screen.queryAllByTestId("waveform-bar").length).toBe(0);
    expect(screen.getByTestId("waveform-empty")).toBeInTheDocument();
  });
});
