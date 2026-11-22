import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initEngine } from "./engine";
import { useDesign } from "./store";
import { DEMO, withoutEngine } from "./demo";
import { readLink } from "./link";
import { useSimulator } from "./simulator/simulatorStore";
import "./styles.css";

if (DEMO) withoutEngine();
await initEngine();
useDesign.getState().bootstrap();
// A link followed inside the page: the view, and in the online demo the simulator's model.
window.addEventListener("hashchange", () => {
  const link = readLink(location.hash);
  if (!link) return;
  useDesign.getState().setView(link.view);
  if (DEMO && link.view === "simulator") void useSimulator.getState().openLink(link.model);
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
