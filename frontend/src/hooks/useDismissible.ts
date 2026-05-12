// 文件说明：该文件属于前端状态 Hook，集中实现 useDismissible 相关逻辑。
import { useEffect, useState } from "react";

const STORAGE_PREFIX = "mmsec:hidden:";
export const DISMISSIBLE_RESET_EVENT = "mmsec:dismissible-reset";

/** 中文注释：实现 storageKeyFor 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
function storageKeyFor(id?: string) {
  return id ? `${STORAGE_PREFIX}${id}` : "";
}

/** 中文注释：实现 readHidden 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
function readHidden(id?: string) {
  if (!id || typeof window === "undefined") {
    return false;
  }
  return window.sessionStorage.getItem(storageKeyFor(id)) === "1";
}

/** 中文注释：实现 clearDismissedPanels 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
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

/** 中文注释：实现 useDismissible 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
export function useDismissible(id?: string) {
  const [visible, setVisible] = useState(() => !readHidden(id));

  useEffect(() => {
    setVisible(!readHidden(id));
  }, [id]);

  useEffect(() => {
    if (!id || typeof window === "undefined") {
      return undefined;
    }
    /** 中文注释：实现 handleReset 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
    const handleReset = () => setVisible(true);
    window.addEventListener(DISMISSIBLE_RESET_EVENT, handleReset);
    return () => window.removeEventListener(DISMISSIBLE_RESET_EVENT, handleReset);
  }, [id]);

  /** 中文注释：实现 dismiss 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
  function dismiss() {
    if (!id || typeof window === "undefined") {
      return;
    }
    window.sessionStorage.setItem(storageKeyFor(id), "1");
    setVisible(false);
  }

  /** 中文注释：实现 restore 的核心流程，支撑前端状态 Hook中的业务语义和异常边界。 */
  function restore() {
    if (!id || typeof window === "undefined") {
      return;
    }
    window.sessionStorage.removeItem(storageKeyFor(id));
    setVisible(true);
  }

  return { visible, dismiss, restore };
}
