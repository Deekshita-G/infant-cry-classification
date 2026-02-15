self.addEventListener('install', function(e) {
  console.log('Service Worker Installed');
});

self.addEventListener('fetch', function(event) {});
self.addEventListener("install",()=>self.skipWaiting());
self.addEventListener("fetch",()=>{});
