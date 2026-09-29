import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser', fullyParallel: false,
  use: {baseURL:'http://localhost:5173', channel:'chrome', permissions:['microphone'], launchOptions:{args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']}, viewport:{width:1440,height:1100}},
  webServer:{command:'npm run dev',url:'http://localhost:5173',reuseExistingServer:true},
});
