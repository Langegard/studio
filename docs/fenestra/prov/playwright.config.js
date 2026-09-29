// @ts-check
const { defineConfig } = require('@playwright/test');
const path = require('path');

const PORT = 4646;

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: /fenestra\.spec\.js/,
  timeout: 30000,
  retries: 0,
  workers: 1,
  reporter: [['./utfall-reporter.js']],
  use: {
    baseURL: 'http://127.0.0.1:' + PORT,
    browserName: 'chromium',
    viewport: { width: 375, height: 812 },
    deviceScaleFactor: 3,
    isMobile: true,
    hasTouch: true,
    locale: 'sv-SE',
    launchOptions: process.env.FENESTRA_CHROMIUM ? { executablePath: process.env.FENESTRA_CHROMIUM } : {}
  },
  webServer: {
    command: 'node ' + path.join(__dirname, '..', 'fixtur', 'server.js'),
    url: 'http://127.0.0.1:' + PORT + '/',
    reuseExistingServer: true,
    env: { PORT: String(PORT) }
  }
});
