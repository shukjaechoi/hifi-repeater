export type Take = { id: string; slot: number; created: number; samples: Float32Array<ArrayBuffer>; rate: number; saved: boolean; start: number; end: number; settings?: MediaTrackSettings };
const db = new Promise<IDBDatabase>((resolve, reject) => {
  const req = indexedDB.open('hifi-repeater', 1);
  req.onupgradeneeded = () => req.result.createObjectStore('takes', {keyPath: 'id'});
  req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error);
});
export async function allTakes(): Promise<Take[]> {
  const database = await db;
  return new Promise((resolve, reject) => {
    const request = database.transaction('takes').objectStore('takes').getAll();
    request.onsuccess = () => resolve(request.result.sort((a: Take, b: Take) => b.created - a.created));
    request.onerror = () => reject(request.error);
  });
}
export async function putTake(take: Take): Promise<void> {
  const database = await db;
  return new Promise((resolve, reject) => {
    const tx = database.transaction('takes', 'readwrite'); tx.objectStore('takes').put(take);
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error);
  });
}
