import { useEffect } from "react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  /** Width class (Tailwind). Default: max-w-3xl. */
  widthClass?: string;
  children: React.ReactNode;
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
        className={`w-full ${widthClass} max-h-[90vh] overflow-y-auto rounded-lg border border-neutral-800 bg-neutral-950 shadow-2xl`}
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
