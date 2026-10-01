import { defineConfig } from '@playwright/test';
const port = process.env.PLAYWRIGHT_PORT || '5173';
export default defineConfig({
  testDir: './tests/browser', fullyParallel: false,
  use: {baseURL:`http://localhost:${port}`, channel:'chrome', permissions:['microphone'], launchOptions:{args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']}, viewport:{width:1440,height:1100}},
  webServer:{command:`npm run dev -- --port ${port}`,url:`http://localhost:${port}`,reuseExistingServer:true},
});
