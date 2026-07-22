// Registers the @material/web (Material 3) custom elements used by the UI.
//
// The app's own CSS classes still drive most layout; @material/web components
// supply the Material 3 interactive primitives (buttons, list, icon buttons,
// navigation rail, switch, elevation). They are themed through the official
// --md-sys-color-* tokens declared in theme.css.
//
// Imported from main.tsx only, so unit tests (which import ./App directly and
// never touch main.tsx) are not loaded into Lit/jsdom and stay isolated.

import "@material/web/button/filled-button.js";
import "@material/web/button/outlined-button.js";
import "@material/web/button/text-button.js";
import "@material/web/iconbutton/icon-button.js";
import "@material/web/iconbutton/outlined-icon-button.js";
import "@material/web/elevation/elevation.js";
import "@material/web/labs/card/elevated-card.js";
import "@material/web/labs/card/filled-card.js";
import "@material/web/labs/card/outlined-card.js";
import "@material/web/list/list.js";
import "@material/web/list/list-item.js";
import "@material/web/focus/md-focus-ring.js";
import "@material/web/ripple/ripple.js";
