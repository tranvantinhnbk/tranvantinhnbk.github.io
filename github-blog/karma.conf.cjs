module.exports = function (config) {
  config.set({
    frameworks: ['jasmine', '@angular-devkit/build-angular'],
    plugins: [
      require('karma-jasmine'),
      require('karma-chrome-launcher'),
      require('@angular-devkit/build-angular/plugins/karma'),
    ],
    reporters: ['progress'],
    hostname: '127.0.0.1',
    listenAddress: '127.0.0.1',
    port: 9876,
    browsers: ['ChromeHeadless'],
    singleRun: true,
    restartOnFileChange: false,
  });
};
