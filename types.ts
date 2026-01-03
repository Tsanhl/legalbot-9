export interface UploadedDocument {
  id: string;
  type: 'file' | 'link';
  name: string;
  mimeType: string;
  data: string; // Base64 string for files, URL string for links
  size: number;
}

export enum MessageRole {
  USER = 'user',
  MODEL = 'model',
  SYSTEM = 'system'
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: number;
  isError?: boolean;
  groundingUrls?: Array<{title: string, uri: string}>;
}

export interface ProcessingState {
  isThinking: boolean;
  stage?: 'uploading' | 'analyzing' | 'citing' | 'complete';
}

export interface Citation {
  ref: string;   // The visible text e.g. [Smith v Jones]
  doc: string;   // Document name
  loc: string;   // Location e.g. Para 45
}

export interface Project {
  id: string;
  name: string;
  messages: ChatMessage[];
  documents: UploadedDocument[];
  createdAt: number;
  updatedAt: number;
  crossMemory: boolean; // Whether to share memory with other projects
}