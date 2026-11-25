import { useEffect, useRef } from "react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  /** Width class (Tailwind). Default: max-w-3xl. */
  widthClass?: string;
  children: React.ReactNode;
}

const FOCUSABLE = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

function focusableIn(dialog: HTMLElement): HTMLElement[] {
  return Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE));
}

function focusIsNowhere(): boolean {
  return document.activeElement == null || document.activeElement === document.body;
}

function keepTabInside(dialog: HTMLElement, e: KeyboardEvent): void {
  const focused = document.activeElement;
  const strayed = focusIsNowhere() || focused === dialog;
  const controls = focusableIn(dialog);
  const first = controls[0] ?? dialog;
  const last = controls[controls.length - 1] ?? dialog;
  const leaving = strayed || focused === (e.shiftKey ? first : last);
  if (!leaving) return;
  e.preventDefault();
  (e.shiftKey ? last : first).focus();
}

function useOpener(open: boolean): React.MutableRefObject<Element | null> {
  const opener = useRef<Element | null>(null);
  const seenOpen = useRef(false);
  // Read while rendering: a child's autoFocus moves focus before any effect here runs.
  if (open !== seenOpen.current) {
    seenOpen.current = open;
    if (open) opener.current = document.activeElement;
  }
  return opener;
}

/** Small dependency-free modal. ESC closes; click on the backdrop closes;
 * inner clicks don't bubble up. Content scrolls if it overflows the viewport. */
export function Modal({
  open,
  onClose,
  title,
  widthClass = "max-w-3xl",
  children,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const opener = useOpener(open);

  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    if (dialog && !dialog.contains(document.activeElement)) {
      (focusableIn(dialog)[0] ?? dialog).focus();
    }
    function onTab(e: KeyboardEvent): void {
      if (e.key === "Tab" && dialogRef.current) keepTabInside(dialogRef.current, e);
    }
    window.addEventListener("keydown", onTab);
    return () => {
      window.removeEventListener("keydown", onTab);
      const cameFrom = opener.current;
      if (cameFrom instanceof HTMLElement && cameFrom.isConnected && focusIsNowhere()) {
        cameFrom.focus();
      }
    };
  }, [open, opener]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent): void {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    // Lock background scroll while open
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center p-6 bg-black/70 backdrop-blur-sm"
      onClick={onClose}
      role="presentation"
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        className={`w-full ${widthClass} max-h-[90vh] overflow-y-auto rounded-lg border border-neutral-800 bg-neutral-950 shadow-2xl outline-none`}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        {title && (
          <div className="sticky top-0 flex items-center justify-between px-5 py-3 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur z-10">
            <h2 className="text-sm font-semibold tracking-wide uppercase text-neutral-200">
              {title}
            </h2>
            <button
              type="button"
              onClick={onClose}
              className="text-neutral-500 hover:text-neutral-200 text-xl leading-none"
              aria-label="Close"
            >
              ×
            </button>
          </div>
        )}
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}
