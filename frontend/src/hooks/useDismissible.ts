// 文件说明：该文件属于前端状态 Hook，集中实现 useDismissible 相关逻辑。
import { useEffect, useState } from "react";

const STORAGE_PREFIX = "mmsec:hidden:";
export const DISMISSIBLE_RESET_EVENT = "mmsec:dismissible-reset";

/** 整理 `storage key 所属` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function storageKeyFor(id?: string) {
  return id ? `${STORAGE_PREFIX}${id}` : "";
}

/** 整理 `read hidden` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function readHidden(id?: string) {
  if (!id || typeof window === "undefined") {
    return false;
  }
  return window.sessionStorage.getItem(storageKeyFor(id)) === "1";
}

/** 整理 `clear dismissed panels` 前端辅助逻辑，保持数据转换和展示口径一致。 */
export function clearDismissedPanels() {
  if (typeof window === "undefined") {
    return;
  }
  const keysToRemove: string[] = [];
  for (let index = 0; index < window.sessionStorage.length; index += 1) {
    const key = window.sessionStorage.key(index);
    if (key?.startsWith(STORAGE_PREFIX)) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => window.sessionStorage.removeItem(key));
  window.dispatchEvent(new Event(DISMISSIBLE_RESET_EVENT));
}

/** 封装 `useDismissible` Hook，把页面状态、副作用和持久化逻辑集中管理。 */
export function useDismissible(id?: string) {
  const [visible, setVisible] = useState(() => !readHidden(id));

  useEffect(() => {
    setVisible(!readHidden(id));
  }, [id]);

  useEffect(() => {
    if (!id || typeof window === "undefined") {
      return undefined;
    }
    /** 处理 `handle reset` 交互事件，把用户操作同步到页面状态。 */
    const handleReset = () => setVisible(true);
    window.addEventListener(DISMISSIBLE_RESET_EVENT, handleReset);
    return () => window.removeEventListener(DISMISSIBLE_RESET_EVENT, handleReset);
  }, [id]);

  /** 整理 `dismiss` 前端辅助逻辑，保持数据转换和展示口径一致。 */
  function dismiss() {
    if (!id || typeof window === "undefined") {
      return;
    }
    window.sessionStorage.setItem(storageKeyFor(id), "1");
    setVisible(false);
  }

  /** 整理 `restore` 前端辅助逻辑，保持数据转换和展示口径一致。 */
  function restore() {
    if (!id || typeof window === "undefined") {
      return;
    }
    window.sessionStorage.removeItem(storageKeyFor(id));
    setVisible(true);
  }

  return { visible, dismiss, restore };
}
