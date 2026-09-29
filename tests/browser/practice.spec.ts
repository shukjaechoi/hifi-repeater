import { test, expect } from '@playwright/test';
test('record → automatic loop → second slot → persistence and WAV export', async ({page}) => {
  const errors: string[]=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  expect(await page.evaluate(() => window.isSecureContext)).toBe(true);
  await page.getByRole('button',{name:'구간 반복 OFF'}).click();
  await page.getByRole('button',{name:'녹음 시작',exact:true}).click();
  await expect(page.getByRole('button',{name:'정지하고 듣기'})).toBeEnabled();
  await page.waitForTimeout(1200);
  await page.getByRole('button',{name:'정지하고 듣기'}).click();
  await expect(page.getByText('LOOPING',{exact:true})).toBeVisible();
  await expect(page.getByText('Take 01',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'재생 정지',exact:true}).click();
  await page.getByRole('button',{name:'앞 무음 제외'}).click();
  await expect(page.getByRole('status')).toContainText('앞 무음');
  await page.getByRole('slider',{name:'구간 시작'}).fill('0.2');
  await page.getByRole('button',{name:'선택 구간 확대'}).click();
  await expect(page.getByRole('button',{name:'전체 보기'})).toBeVisible();
  await page.getByRole('button',{name:'따로 보관'}).click();
  const downloadPromise=page.waitForEvent('download');
  await page.getByRole('button',{name:'WAV 내보내기'}).click();
  const download=await downloadPromise; expect(download.suggestedFilename()).toMatch(/\.wav$/);
  await page.getByRole('button',{name:/SLOT 02/}).click();
  await expect(page.getByText('아직 녹음된 연주가 없습니다.',{exact:false})).toBeVisible();
  await page.getByRole('button',{name:'녹음 시작',exact:true}).click();
  await expect(page.getByRole('button',{name:'정지하고 듣기'})).toBeEnabled();
  await page.waitForTimeout(500);
  await page.getByRole('button',{name:'정지하고 듣기'}).click();
  await expect(page.getByText('LOOPING',{exact:true})).toBeVisible();
  await page.reload();
  await expect(page.getByText('Take 01 ★',{exact:true})).toBeVisible();
  await expect(page.getByRole('slider',{name:'구간 시작'})).toHaveValue('0.2');
  await page.screenshot({path:'test-results/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.reload();
  await expect(page.getByText('Take 01 ★',{exact:true})).toBeVisible();
  await page.screenshot({path:'test-results/mobile.png',fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});
test('microphone denial returns to a usable idle state', async ({browser})=>{
  const context=await browser.newContext({permissions:[]});const page=await context.newPage();
  await page.addInitScript(()=>{ navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Microphone denied','NotAllowedError');}; });
  await page.goto('http://localhost:5173');
  await page.getByRole('button',{name:'녹음 시작',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Microphone denied');
  await expect(page.getByRole('button',{name:'녹음 시작',exact:true})).toBeEnabled();
  await context.close();
});

test('default automatic playback ends and a new take preserves the previous one', async ({page}) => {
  await page.goto('/');
  await expect(page.getByRole('button',{name:'구간 반복 OFF'})).toBeVisible();
  for (let i=1;i<=2;i++) {
    await page.getByRole('button',{name:'녹음 시작',exact:true}).click();
    await expect(page.getByRole('button',{name:'정지하고 듣기'})).toBeEnabled();
    await expect(page.getByRole('button',{name:/SLOT 02/})).toBeDisabled();
    await page.waitForTimeout(900);
    await page.getByRole('button',{name:'정지하고 듣기'}).click();
    await expect(page.getByText('PLAYING',{exact:true})).toBeVisible();
    await expect(page.getByText('READY TO LISTEN',{exact:true})).toBeVisible();
    await expect(page.getByText(`Take 0${i}`,{exact:true})).toBeVisible();
  }
  await expect(page.getByText('Take 01',{exact:true})).toBeVisible();
  await expect(page.getByText('Take 02',{exact:true})).toBeVisible();
});
