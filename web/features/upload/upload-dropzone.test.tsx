import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";

import { ToastProvider } from "@/components/toast-provider";
import { UploadDropzone } from "@/features/upload/upload-dropzone";

class MockXMLHttpRequest {
  static instances: MockXMLHttpRequest[] = [];

  status = 200;
  responseText = '{"documents":[{"id":"doc-1"}]}';
  upload = { onprogress: null as ((event: ProgressEvent<EventTarget>) => void) | null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  method = "";
  url = "";

  constructor() {
    MockXMLHttpRequest.instances.push(this);
  }

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  send() {
    this.upload.onprogress?.({ lengthComputable: true, loaded: 10, total: 10 } as ProgressEvent<EventTarget>);
    this.onload?.();
  }
}

test("uploads a file against the versioned backend endpoint", async () => {
  const originalXHR = global.XMLHttpRequest;
  global.XMLHttpRequest = MockXMLHttpRequest as unknown as typeof XMLHttpRequest;
  const onUploaded = vi.fn();

  render(
    <ToastProvider>
      <UploadDropzone onUploaded={onUploaded} />
    </ToastProvider>
  );

  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  const file = new File(["hello"], "guide.txt", { type: "text/plain" });
  fireEvent.change(input, { target: { files: [file] } });

  await waitFor(() => {
    expect(MockXMLHttpRequest.instances[0]?.url).toMatch(/\/api\/v1\/documents\/upload$/);
  });
  expect(await screen.findByText(/guide.txt uploaded/i)).toBeInTheDocument();
  expect(onUploaded).toHaveBeenCalled();

  global.XMLHttpRequest = originalXHR;
});
