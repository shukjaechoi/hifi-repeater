import {test,expect} from '@playwright/test';
test('register music, toggle takes with one live mic, persist and release',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{
    let context:AudioContext, output:MediaStreamAudioDestinationNode;
    const harness={opens:0, stream:undefined as MediaStream|undefined, sound:async()=>{
      await context.resume();
      const start=context.currentTime+.05;
      [330,392,523].forEach((frequency,i)=>{
        const oscillator=context.createOscillator(),gain=context.createGain();oscillator.frequency.value=frequency;
        gain.gain.setValueAtTime(0,start+i*.24);gain.gain.linearRampToValueAtTime(.3,start+i*.24+.01);gain.gain.linearRampToValueAtTime(0,start+(i+1)*.24);
        oscillator.connect(gain).connect(output);oscillator.start(start+i*.24);oscillator.stop(start+(i+1)*.24);
      });
    }};
    (window as any).musicHarness=harness;
    navigator.mediaDevices.getUserMedia=async()=>{
      harness.opens++;context??=new AudioContext();output=context.createMediaStreamDestination();harness.stream=output.stream;return output.stream;
    };
  });
  await page.goto('/');await page.getByRole('tab',{name:'설정',exact:true}).click();
  await expect(page.getByRole('checkbox',{name:'Trigger sound 활성화'})).toBeDisabled();
  await page.getByRole('button',{name:'트리거 소리 녹음하기'}).click();
  await expect(page.getByRole('button',{name:'등록 녹음 마치기'})).toBeVisible();
  await page.waitForTimeout(350);await page.evaluate(()=>(window as any).musicHarness.sound());await page.waitForTimeout(1100);
  await page.getByRole('button',{name:'등록 녹음 마치기'}).click();
  await expect(page.getByText('등록됨 · 활성화하면 마이크를 엽니다.',{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>(window as any).musicHarness.stream.active)).toBe(false);
  await page.getByRole('checkbox',{name:'Trigger sound 활성화'}).check();
  await page.getByRole('tab',{name:'연습',exact:true}).click();
  await page.waitForTimeout(400);await page.evaluate(()=>(window as any).musicHarness.sound());
  await expect(page.getByRole('button',{name:'정지하고 듣기',exact:true})).toBeVisible();
  await page.waitForTimeout(1500);await page.evaluate(()=>(window as any).musicHarness.sound());
  await expect(page.getByText('Take 01',{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>(window as any).musicHarness.opens)).toBe(2);
  expect(await page.evaluate(()=>(window as any).musicHarness.stream.active)).toBe(true);
  await page.waitForTimeout(2200);
  await expect(page.getByRole('button',{name:'녹음 시작',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'녹음 시작',exact:true}).click();
  await expect(page.getByRole('button',{name:'정지하고 듣기',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'끄기',exact:true}).click();
  expect(await page.evaluate(()=>(window as any).musicHarness.stream.active)).toBe(true);
  await page.waitForTimeout(300);
  await page.getByRole('button',{name:'정지하고 듣기',exact:true}).click();
  await expect(page.getByText('Take 02',{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>(window as any).musicHarness.stream.active)).toBe(false);
  await page.reload();await page.getByRole('tab',{name:'설정',exact:true}).click();
  await expect(page.getByRole('checkbox',{name:'Trigger sound 활성화'})).toBeEnabled();
  await expect(page.getByRole('checkbox',{name:'Trigger sound 활성화'})).not.toBeChecked();
  await page.getByRole('button',{name:'트리거 삭제',exact:true}).click();
  await expect(page.getByRole('checkbox',{name:'Trigger sound 활성화'})).toBeDisabled();
  expect(errors).toEqual([]);
  await page.screenshot({path:'test-results/trigger-settings.png',fullPage:true});
});

test('registration cancellation and denied microphone leave usable controls',async({page})=>{
  await page.goto('/');await page.getByRole('tab',{name:'설정',exact:true}).click();
  await page.getByRole('button',{name:'트리거 소리 녹음하기'}).click();
  await expect(page.getByRole('button',{name:'등록 녹음 마치기'})).toBeVisible();
  await page.getByRole('button',{name:'등록 취소'}).click();
  await expect(page.getByRole('checkbox',{name:'Trigger sound 활성화'})).toBeDisabled();
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new Error('Microphone denied');};});
  await page.getByRole('button',{name:'트리거 소리 녹음하기'}).click();
  await expect(page.getByRole('alert')).toContainText('Microphone denied');
  await expect(page.getByRole('button',{name:'트리거 소리 녹음하기'})).toBeEnabled();
  await page.getByRole('tab',{name:'연습',exact:true}).click();
  await expect(page.getByRole('button',{name:'녹음 시작',exact:true})).toBeEnabled();
});
