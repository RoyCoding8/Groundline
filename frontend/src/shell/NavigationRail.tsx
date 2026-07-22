import {
  ChartLine,
  Flask,
  House,
  List,
  Moon,
  PencilSimple,
  Sun,
} from "@phosphor-icons/react";

export type View = "dashboard" | "experiments" | "builder" | "run";
export type BackendStatus = "checking" | "online" | "offline";
export type ThemeChoice = "" | "light" | "dark";

type NavigationRailProps = {
  view: View;
  onNavigate: (view: View) => void;
  backendStatus: BackendStatus;
  theme: ThemeChoice;
  resolvedDark: boolean;
  onToggleTheme: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
};

const items: Array<{ view: View; label: string; icon: typeof House }> = [
  { view: "dashboard", label: "Dashboard", icon: House },
  { view: "experiments", label: "Experiments", icon: Flask },
  { view: "builder", label: "Builder", icon: PencilSimple },
  { view: "run", label: "Run Viewer", icon: ChartLine },
];

export function NavigationRail({
  view,
  onNavigate,
  backendStatus,
  theme,
  resolvedDark,
  onToggleTheme,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}: NavigationRailProps) {
  const statusLabel = backendStatus === "online" ? "Connected" : backendStatus === "offline" ? "Offline" : "Checking...";
  const nextLabel = theme === "light" ? "dark" : theme === "dark" ? "system" : "light";
  const modeLabel = theme === "light" ? "Light" : theme === "dark" ? "Dark" : "System";

  return (
    <>
      <div
        className={`sidebar-backdrop ${mobileOpen ? "visible" : ""}`}
        onClick={onCloseMobile}
        onKeyDown={(event) => { if (event.key === "Escape") onCloseMobile(); }}
        role="button"
        tabIndex={-1}
        aria-label="Close sidebar"
      />
      <nav className={`sidebar ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`} aria-label="Main navigation">
        <div className="sidebar-header">
          <span className="sidebar-brand-mark" aria-hidden="true"><span /></span>
          <span className="sidebar-title">Groundline</span>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={onToggleCollapse}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <List size={18} weight="bold" />
          </button>
        </div>

        <div className="sidebar-nav">
          {items.map((item) => {
            const Icon = item.icon;
            const active = view === item.view;
            return (
              <button
                key={item.view}
                type="button"
                className={`sidebar-nav-item ${active ? "active" : ""}`}
                onClick={() => { onNavigate(item.view); onCloseMobile(); }}
                aria-label={item.label}
                aria-current={active ? "page" : undefined}
              >
                <span className="nav-icon"><Icon size={21} weight={active ? "fill" : "regular"} /></span>
                <span className="nav-label">{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <span className={`status-dot ${backendStatus}`} />
            <span>{statusLabel}</span>
          </div>
          <button
            type="button"
            className="sidebar-nav-item theme-rail-button"
            onClick={onToggleTheme}
            aria-label={`Switch to ${nextLabel} mode`}
          >
            <span className="nav-icon">{resolvedDark ? <Moon size={20} /> : <Sun size={20} />}</span>
            <span className="nav-label">{modeLabel}</span>
          </button>
        </div>
      </nav>
    </>
  );
}
