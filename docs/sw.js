// Cache de l'application (fonctionne hors ligne avec les dernières données reçues)
const CACHE = "signal-ia-v3";
const SHELL = ["./", "index.html", "manifest.webmanifest", "icon-192.png", "icon-512.png", "apple-touch-icon.png"];
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  // Réseau d'abord (données et page toujours à jour), cache si hors ligne
  e.respondWith(fetch(e.request).then(r => {
    const copy = r.clone();
    const key = url.pathname.endsWith(".json") ? new Request(url.origin + url.pathname) : e.request;
    caches.open(CACHE).then(c => c.put(key, copy));
    return r;
  }).catch(() => caches.match(url.pathname.endsWith(".json") ? new Request(url.origin + url.pathname) : e.request)));
});
