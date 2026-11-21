import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initEngine } from "./engine";
import { useDesign } from "./store";
import { DEMO, withoutEngine } from "./demo";
import "./styles.css";

if (DEMO) withoutEngine();
await initEngine();
useDesign.getState().bootstrap();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
