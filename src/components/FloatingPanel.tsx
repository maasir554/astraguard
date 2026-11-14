import { useId, useState, type ReactNode, type PointerEvent } from "react";
import { Minus, PanelLeftOpen, PanelRightOpen, RotateCcw } from "lucide-react";
import { Button } from "./ui/button";

interface Props {
  title: string;
  side: "left" | "right";
  children: ReactNode;
}

/** Viewport-bounded window. Its body scrolls independently of the map and header. */
export function FloatingPanel({ title, side, children }: Props) {
  const id = useId();
  const [minimized, setMinimized] = useState(
    () => side === "right" && window.innerWidth < 900,
  );
  const [size, setSize] = useState({
    width: side === "left" ? 310 : 340,
    height: 720,
  });
  const Icon = side === "left" ? PanelLeftOpen : PanelRightOpen;
  function resize(event: PointerEvent<HTMLButtonElement>) {
    const handle = event.currentTarget;
    const rect = handle.closest("section")!.getBoundingClientRect();
    const start = {
      x: event.clientX,
      y: event.clientY,
      width: rect.width,
      height: rect.height,
    };
    handle.setPointerCapture(event.pointerId);
    function move(e: globalThis.PointerEvent) {
      setSize({
        width: Math.max(
          270,
          Math.min(
            window.innerWidth - 32,
            start.width + (e.clientX - start.x) * (side === "left" ? 1 : -1),
          ),
        ),
        height: Math.max(
          220,
          Math.min(
            window.innerHeight - 112,
            start.height + e.clientY - start.y,
          ),
        ),
      });
    }
    function end() {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
    }
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  }
  return (
    <section
      className={`floating-panel floating-${side} ${minimized ? "is-minimized" : ""}`}
      aria-label={title}
      style={{
        width: minimized ? undefined : size.width,
        height: minimized ? undefined : size.height,
      }}
    >
      <div className="floating-titlebar">
        <Icon size={16} />
        <h2>{title}</h2>
        {!minimized && (
          <Button
            variant="ghost"
            size="icon"
            aria-label={`Reset ${title.toLowerCase()} size`}
            onClick={() =>
              setSize({ width: side === "left" ? 310 : 340, height: 720 })
            }
          >
            <RotateCcw />
          </Button>
        )}
        <Button
          variant="ghost"
          size="icon"
          aria-label={`${minimized ? "Restore" : "Minimize"} ${title.toLowerCase()}`}
          aria-expanded={!minimized}
          aria-controls={id}
          onClick={() => setMinimized(!minimized)}
        >
          {minimized ? <Icon /> : <Minus />}
        </Button>
      </div>
      <div id={id} className="floating-body" hidden={minimized}>
        {children}
      </div>
      {!minimized && (
        <button
          className="panel-resize"
          aria-label={`Resize ${title.toLowerCase()}`}
          title="Drag to resize. Use arrow keys when focused."
          onPointerDown={resize}
          onKeyDown={(e) => {
            if (
              !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(
                e.key,
              )
            )
              return;
            e.preventDefault();
            setSize((s) => ({
              width: Math.max(
                270,
                Math.min(
                  window.innerWidth - 32,
                  s.width +
                    (e.key === "ArrowRight"
                      ? 20
                      : e.key === "ArrowLeft"
                        ? -20
                        : 0),
                ),
              ),
              height: Math.max(
                220,
                Math.min(
                  window.innerHeight - 112,
                  s.height +
                    (e.key === "ArrowDown"
                      ? 20
                      : e.key === "ArrowUp"
                        ? -20
                        : 0),
                ),
              ),
            }));
          }}
        >
          <span>⋰</span>
        </button>
      )}
    </section>
  );
}
