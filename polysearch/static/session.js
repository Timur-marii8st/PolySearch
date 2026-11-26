"use strict";
// Same-origin browser client; the server enforces RBAC regardless of the UI.
(() => {
  const original = window.fetch.bind(window);
  window.fetch = async (resource, options={}) => {
    const url = new URL(typeof resource === "string" ? resource : resource.url, location.href);
    if (url.origin !== location.origin) return original(resource,options);
    const method = (options.method || "GET").toUpperCase();
    const headers = new Headers(options.headers || {});
    if (!["GET","HEAD","OPTIONS"].includes(method)) headers.set("X-PolySearch-Request","1");
    const r = await original(resource,{...options,headers});
    if (r.status===401 && url.pathname!=="/api/auth/login") location.replace("/login");
    return r;
  };
  async function session() {
    try {
      const r=await fetch("/api/auth/me");
      if(!r.ok)return;
      const info=await r.json();
      if(!info.authenticated_mode)return;
      const box=document.createElement("div");box.className="session-panel";
      const title=document.createElement("span");title.textContent=info.user.username+" · "+info.user.role;
      const logout=document.createElement("button");logout.type="button";logout.className="button secondary";logout.textContent="Выйти";
      logout.addEventListener("click",async()=>{const r=await fetch("/api/auth/logout",{method:"POST"});if(r.ok)location.replace("/login");});
      box.append(title,logout);document.querySelector(".sidebar").append(box);
      const update=()=>{document.querySelectorAll('input[name="actor"]').forEach(i=>{if(i.value!==info.user.username)i.value=info.user.username;i.readOnly=true;});};
      update();new MutationObserver(update).observe(document.body,{childList:true,subtree:true});
    } catch(e) { console.error("Session status unavailable"); }
  }
  document.addEventListener("DOMContentLoaded",session);
})();
