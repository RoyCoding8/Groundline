// Ambient declarations for the @material/web (Material 3) custom elements used
// in JSX. Lit registers these as Custom Elements; from React's JSX perspective
// they are host elements. React 19's react-jsx runtime resolves intrinsic
// elements through React.JSX.IntrinsicElements (the JSX namespace exported by
// the "react" module), so we augment that. This file is a module so the
// augmentation merges with React's own types instead of replacing them.

import type { DetailedHTMLProps, HTMLAttributes } from "react";

type MdElement = DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement>;

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      "md-elevated-card": MdElement;
      "md-filled-card": MdElement;
      "md-outlined-card": MdElement;
      "md-filled-button": MdElement;
      "md-outlined-button": MdElement;
      "md-text-button": MdElement;
      "md-icon-button": MdElement;
      "md-outlined-icon-button": MdElement;
      "md-list": MdElement;
      "md-list-item": MdElement;
      "md-elevation": MdElement;
      "md-ripple": MdElement;
    }
  }
}

export {};
