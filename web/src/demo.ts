/** The online demo: the interface alone, built with VITE_DEMO=1 and served as static files.
 * Everything that runs on WebAssembly works; what needs a model's weights needs the local engine. */
export const DEMO = import.meta.env.VITE_DEMO === "1";

export const REPOSITORY = "https://github.com/alvarograciagil/llm2chip";
export const INSTALL_GUIDE = `${REPOSITORY}/blob/main/docs/install.md`;

/** There is no engine behind the demo: its calls fail at once, as when it is not running. */
export function withoutEngine() {
  const pass = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = new URL(input instanceof Request ? input.url : String(input), location.href);
    return url.origin === location.origin && url.pathname.startsWith("/api/") ? Promise.reject(new TypeError()) : pass(input, init);
  };
}
