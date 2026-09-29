import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';
export default defineConfig(({mode}) => {
  const https = mode === 'https' ? {
    key: readFileSync(new URL('./.certs/localhost-key.pem', import.meta.url)),
    cert: readFileSync(new URL('./.certs/localhost.pem', import.meta.url)),
  } : undefined;
  return {base:process.env.VITE_BASE_PATH || '/',server:{host:'0.0.0.0',port:5173,strictPort:true,https},preview:{host:'0.0.0.0',port:4173,strictPort:true,https}};
});
