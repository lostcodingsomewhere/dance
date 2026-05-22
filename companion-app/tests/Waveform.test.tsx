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

  it("renders section bands and cue ticks from regions", () => {
    render(
      <Waveform
        peaks={[0.1, 0.5, 0.9, 0.2]}
        durationSeconds={100}
        regions={[
          {
            id: 1,
            position_ms: 0,
            length_ms: 30000,
            region_type: "section",
            section_label: "intro",
            length_bars: null,
            name: null,
            color: null,
            confidence: null,
            source: "test",
            stem_file_id: null,
          },
          {
            id: 2,
            position_ms: 30000,
            length_ms: null,
            region_type: "cue",
            section_label: null,
            length_bars: null,
            name: null,
            color: null,
            confidence: null,
            source: "test",
            stem_file_id: null,
          },
        ]}
      />,
    );
    const sections = screen.getAllByTestId("waveform-section");
    expect(sections.length).toBe(1);
    // 30 s / 100 s = 30 % of viewBox width
    expect(sections[0].getAttribute("x")).toBe("0");
    expect(sections[0].getAttribute("width")).toBe("30");
    const ticks = screen.getAllByTestId("waveform-cue");
    expect(ticks.length).toBe(1);
    expect(ticks[0].getAttribute("x1")).toBe("30");
  });

  it("omits region overlays when durationSeconds is missing", () => {
    render(
      <Waveform
        peaks={[0.1, 0.5]}
        regions={[
          {
            id: 1,
            position_ms: 0,
            length_ms: 30000,
            region_type: "section",
            section_label: "drop",
            length_bars: null,
            name: null,
            color: null,
            confidence: null,
            source: "test",
            stem_file_id: null,
          },
        ]}
      />,
    );
    expect(screen.queryAllByTestId("waveform-section").length).toBe(0);
  });
});
