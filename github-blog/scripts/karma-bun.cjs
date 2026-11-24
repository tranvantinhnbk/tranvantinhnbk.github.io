// Karma 6 uses comment-based DI and passes a pre-bound net.Server to HTTP.listen.
// Bun strips those comments and does not accept that server-handle handoff.
// Keep the browser tests intact by supplying explicit DI and a normal HTTP bind.
module.exports = function configureBun(config) {
  if (!process.versions.bun) return;
  require('karma/lib/middleware/proxy').create.$inject = ['config', 'config.proxies', 'emitter'];
  require('karma/lib/browser').factory.$inject = [
    'id', 'fullName', 'capturedBrowsers', 'emitter', 'socket', 'timer',
    'config.browserDisconnectTimeout', 'config.browserNoActivityTimeout',
    'config.singleRun', 'config.client',
  ];
  const http = require('node:http');
  config.set({
    httpModule: {
      createServer(...args) {
        const server = http.createServer(...args);
        const listen = server.listen.bind(server);
        server.listen = function (boundServer, callback) {
          const { port, address } = boundServer.address();
          boundServer.close(() => listen(port, address, callback));
          return server;
        };
        return server;
      },
    },
  });
};
