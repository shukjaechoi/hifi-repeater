import { existsSync, mkdirSync, readFileSync, writeFileSync, chmodSync } from 'node:fs';
import { networkInterfaces } from 'node:os';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

// Never install a CA or change the system trust store automatically.
// If the user has already run `mkcert -install`, use its local CA for new
// certificates; otherwise fall back to a browser-untrusted self-signed cert.
const dir = resolve('.certs');
const key = resolve(dir, 'localhost-key.pem');
const cert = resolve(dir, 'localhost.pem');
const names = new Set(['localhost']);
const ips = new Set(['127.0.0.1', '::1']);
for (const addresses of Object.values(networkInterfaces())) {
  for (const address of addresses ?? []) {
    if (address.family === 'IPv4' && !address.internal) ips.add(address.address);
  }
}
const hosts = [...names, ...ips];
const meta = resolve(dir, 'hosts.json');
if (existsSync(key) && existsSync(cert)) {
  if (existsSync(meta)) {
    const oldHosts = JSON.parse(readFileSync(meta, 'utf8'));
    const missing = hosts.filter(host => !oldHosts.includes(host));
    if (missing.length) console.log(`새 네트워크 주소: ${missing.join(', ')}. 휴대폰 접속용 인증서를 갱신하려면 .certs의 인증서 두 파일을 삭제하고 다시 실행하세요.`);
  }
  console.log('HTTPS 인증서 준비됨: https://localhost:5173');
  process.exit(0);
}
mkdirSync(dir, {recursive: true, mode: 0o700});
const mkcert = spawnSync('mkcert', ['-version'], {encoding: 'utf8'});
if (mkcert.status === 0) {
  const result = spawnSync('mkcert', [
    '-key-file', key,
    '-cert-file', cert,
    ...hosts,
  ], {encoding: 'utf8'});
  if (result.status !== 0) {
    console.error(result.stderr || 'mkcert 인증서를 생성할 수 없습니다.');
    process.exit(1);
  }
  chmodSync(key, 0o600);
  writeFileSync(meta, JSON.stringify(hosts, null, 2));
  console.log(`mkcert HTTPS 인증서 준비됨: ${hosts.filter(h => h !== '::1').map(host => `https://${host}:5173`).join(', ')}`);
  process.exit(0);
}
const config = resolve(dir, 'openssl.cnf');
writeFileSync(config, `[req]\ndistinguished_name=dn\nx509_extensions=extensions\nprompt=no\n[dn]\nCN=localhost\n[extensions]\nsubjectAltName=@names\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n[names]\n${[...names].map((host,i)=>`DNS.${i+1}=${host}`).join('\n')}\n${[...ips].map((host,i)=>`IP.${i+1}=${host}`).join('\n')}\n`);
const result = spawnSync('openssl', ['req','-x509','-newkey','rsa:2048','-nodes','-days','30','-keyout',key,'-out',cert,'-config',config], {encoding:'utf8'});
if (result.status !== 0) { console.error(result.stderr || 'OpenSSL을 실행할 수 없습니다.'); process.exit(1); }
chmodSync(key, 0o600);
writeFileSync(meta, JSON.stringify(hosts, null, 2));
console.log('개발용 자체 서명 인증서를 생성했습니다(30일 유효). 최초 접속 시 브라우저 인증서 확인이 필요합니다.');
console.log('시스템 인증서 신뢰 설정은 변경하지 않았습니다. 인증서와 개인 키는 Git에서 제외됩니다.');
console.log(`접속 주소: ${hosts.filter(h=>h !== '::1').map(h=>`https://${h}:5173`).join(', ')}`);
