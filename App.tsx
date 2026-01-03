import React, { useState, useRef, useEffect, useCallback } from 'react';
import { DocumentList } from './components/DocumentList';
import { ChatBubble } from './components/ChatBubble';
import { Button } from './components/Button';
import { UploadedDocument, ChatMessage, MessageRole, Citation, Project } from './types';
import { sendMessageWithDocs, resetSession, initializeKnowledgeBase } from './services/geminiService';
import { loadLawResourceIndex } from './services/knowledgeBaseService';

const MAX_PROJECTS = 10;

const createNewProject = (name?: string): Project => ({
  id: Math.random().toString(36).substr(2, 9),
  name: name || `Project ${new Date().toLocaleDateString()}`,
  messages: [],
  documents: [],
  createdAt: Date.now(),
  updatedAt: Date.now(),
  crossMemory: false,
});

const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProjectId, setCurrentProjectId] = useState<string>('');
  const [inputValue, setInputValue] = useState('');
  const [linkInput, setLinkInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [knowledgeBaseStatus, setKnowledgeBaseStatus] = useState<{loaded: boolean; count: number; categories: string[]}>({loaded: false, count: 0, categories: []});
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // Get current project
  const currentProject = projects.find(p => p.id === currentProjectId);
  const messages = currentProject?.messages || [];
  const documents = currentProject?.documents || [];

  // Load projects and API key from localStorage on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('gemini_api_key');
    if (savedKey) {
        setApiKey(savedKey);
    }
    
    // Load saved projects
    const savedProjects = localStorage.getItem('legal_ai_projects');
    if (savedProjects) {
      try {
        const parsed = JSON.parse(savedProjects);
        if (parsed.length > 0) {
          setProjects(parsed);
          setCurrentProjectId(parsed[0].id);
        } else {
          // Create default project
          const defaultProject = createNewProject('Default Project');
          setProjects([defaultProject]);
          setCurrentProjectId(defaultProject.id);
        }
      } catch {
        const defaultProject = createNewProject('Default Project');
        setProjects([defaultProject]);
        setCurrentProjectId(defaultProject.id);
      }
    } else {
      // Create default project
      const defaultProject = createNewProject('Default Project');
      setProjects([defaultProject]);
      setCurrentProjectId(defaultProject.id);
    }
  }, []);

  // Save projects to localStorage whenever they change
  useEffect(() => {
    if (projects.length > 0) {
      localStorage.setItem('legal_ai_projects', JSON.stringify(projects));
    }
  }, [projects]);
  
  // Initialize knowledge base on mount - this makes Law Resources automatic
  useEffect(() => {
    loadLawResourceIndex().then(index => {
      if (index) {
        setKnowledgeBaseStatus({ 
          loaded: true, 
          count: index.totalFiles,
          categories: index.categories 
        });
        initializeKnowledgeBase();
        console.log('📚 Knowledge Base Loaded:', index.totalFiles, 'documents across', index.categories.length, 'categories');
      }
    });
  }, []);

  // Save API key when changed
  const handleApiKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const newKey = e.target.value;
      setApiKey(newKey);
      localStorage.setItem('gemini_api_key', newKey);
  };

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Helper to read a single File object to Base64
  const readFileToBase64 = (file: File): Promise<UploadedDocument> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          resolve({
            id: Math.random().toString(36).substr(2, 9),
            type: 'file',
            name: file.name,
            mimeType: file.type || 'application/octet-stream',
            data: reader.result.split(',')[1], // Remove data URL prefix
            size: file.size
          });
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const processFiles = async (files: File[]) => {
    // Filter out hidden files or non-relevant types if needed
    const validFiles = files.filter(f => !f.name.startsWith('.'));
    const limit = Math.min(validFiles.length, 50); // Safety limit
    
    const newDocs: UploadedDocument[] = [];
    
    for (let i = 0; i < limit; i++) {
        try {
            const doc = await readFileToBase64(validFiles[i]);
            newDocs.push(doc);
        } catch (err) {
            console.error("Error reading file", validFiles[i].name, err);
        }
    }
    
    setProjects(prev => prev.map(p => 
      p.id === currentProjectId 
        ? { ...p, documents: [...p.documents, ...newDocs], updatedAt: Date.now() }
        : p
    ));
  };

  // Recursive directory scanner for Drop events
  const scanFiles = async (item: any): Promise<File[]> => {
    if (item.isFile) {
        return new Promise((resolve) => {
            item.file((file: File) => resolve([file]));
        });
    } else if (item.isDirectory) {
        const dirReader = item.createReader();
        return new Promise((resolve) => {
            dirReader.readEntries(async (entries: any[]) => {
                const files = await Promise.all(entries.map(scanFiles));
                resolve(files.flat());
            });
        });
    }
    return [];
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const items = e.dataTransfer.items;
    if (!items) return;

    const files: File[] = [];
    
    // Handle modern directory traversal
    const promises = [];
    for (let i = 0; i < items.length; i++) {
        const item = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
        if (item) {
            promises.push(scanFiles(item));
        } else if (items[i].kind === 'file') {
            const file = items[i].getAsFile();
            if (file) files.push(file);
        }
    }

    const results = await Promise.all(promises);
    const allFiles = [...files, ...results.flat()];
    
    if (allFiles.length > 0) {
        await processFiles(allFiles);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only leave if we are leaving the main container, not child elements
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    await processFiles(Array.from(event.target.files));
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFolderUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    await processFiles(Array.from(event.target.files));
    if (folderInputRef.current) folderInputRef.current.value = '';
  };

  const handleAddLink = () => {
    if (!linkInput.trim()) return;
    
    // Basic validation
    let url = linkInput.trim();
    if (!url.startsWith('http')) {
        url = 'https://' + url;
    }

    const newLink: UploadedDocument = {
        id: Math.random().toString(36).substr(2, 9),
        type: 'link',
        name: url,
        mimeType: 'text/uri-list',
        data: url,
        size: 0
    };

    setProjects(prev => prev.map(p => 
      p.id === currentProjectId 
        ? { ...p, documents: [...p.documents, newLink], updatedAt: Date.now() }
        : p
    ));
    setLinkInput('');
  };

  const handleRemoveDocument = (id: string) => {
    setProjects(prev => prev.map(p => 
      p.id === currentProjectId 
        ? { ...p, documents: p.documents.filter(doc => doc.id !== id), updatedAt: Date.now() }
        : p
    ));
  };

  const handleSendMessage = useCallback(async () => {
    if (!inputValue.trim() || !currentProjectId) return;
    
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      role: MessageRole.USER,
      text: inputValue,
      timestamp: Date.now()
    };

    // Add user message to current project
    setProjects(prev => prev.map(p => 
      p.id === currentProjectId 
        ? { ...p, messages: [...p.messages, newMessage], updatedAt: Date.now() }
        : p
    ));
    setInputValue('');
    setIsLoading(true);
    setActiveCitation(null);

    try {
      const response = await sendMessageWithDocs(apiKey, newMessage.text, documents, currentProjectId);
      
      const botMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: MessageRole.MODEL,
        text: response.text,
        timestamp: Date.now(),
        groundingUrls: response.groundingUrls
      };
      
      // Add bot response to current project
      setProjects(prev => prev.map(p => 
        p.id === currentProjectId 
          ? { ...p, messages: [...p.messages, botMessage], updatedAt: Date.now() }
          : p
      ));

    } catch (error: any) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: MessageRole.MODEL,
        text: error.message || "I encountered an error processing your request.",
        timestamp: Date.now(),
        isError: true
      };
      setProjects(prev => prev.map(p => 
        p.id === currentProjectId 
          ? { ...p, messages: [...p.messages, errorMessage], updatedAt: Date.now() }
          : p
      ));
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, documents, apiKey, currentProjectId]);

  const handleReset = () => {
    // Clear current project's messages
    setProjects(prev => prev.map(p => 
      p.id === currentProjectId 
        ? { ...p, messages: [], updatedAt: Date.now() }
        : p
    ));
    resetSession(currentProjectId);
    setActiveCitation(null);
  };

  // Project management functions
  const handleCreateProject = () => {
    if (projects.length >= MAX_PROJECTS) {
      alert(`Maximum ${MAX_PROJECTS} projects allowed. Please delete a project first.`);
      return;
    }
    const newProject = createNewProject();
    setProjects(prev => [newProject, ...prev]);
    setCurrentProjectId(newProject.id);
  };

  const handleDeleteProject = (projectId: string) => {
    if (projects.length === 1) {
      alert('Cannot delete the last project.');
      return;
    }
    setProjects(prev => prev.filter(p => p.id !== projectId));
    if (currentProjectId === projectId) {
      setCurrentProjectId(projects.find(p => p.id !== projectId)?.id || '');
    }
    resetSession(projectId);
  };

  const handleRenameProject = (projectId: string, newName: string) => {
    setProjects(prev => prev.map(p => 
      p.id === projectId 
        ? { ...p, name: newName, updatedAt: Date.now() }
        : p
    ));
    setEditingProjectId(null);
  };

  const handleToggleCrossMemory = (projectId: string) => {
    setProjects(prev => prev.map(p => 
      p.id === projectId 
        ? { ...p, crossMemory: !p.crossMemory, updatedAt: Date.now() }
        : p
    ));
  };

  const handleSwitchProject = (projectId: string) => {
    setCurrentProjectId(projectId);
    setActiveCitation(null);
  };

  // Helper to determine if we should hide location
  const shouldHideLocation = (docName: string) => {
    const lowerName = docName.toLowerCase();
    return lowerName.includes('google search') || 
           lowerName.includes('general authority') || 
           lowerName.includes('internal knowledge') ||
           lowerName.includes('general knowledge');
  };

  return (
    <div className="flex h-screen bg-legal-50 font-sans overflow-hidden">
      {/* Sidebar - Documents */}
      <div 
        className={`${isSidebarOpen ? 'w-80 translate-x-0' : 'w-0 -translate-x-full'} bg-legal-900 text-white flex-shrink-0 transition-all duration-300 ease-in-out flex flex-col border-r border-legal-800 z-20 relative`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drag Overlay */}
        {isDragging && (
            <div className="absolute inset-0 bg-yellow-500 bg-opacity-90 z-50 flex items-center justify-center border-4 border-white border-dashed m-2 rounded-lg">
                <div className="text-center text-legal-900">
                    <svg className="w-16 h-16 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    <p className="font-bold text-xl">Drop "Law Resources" Here</p>
                    <p className="text-sm">I'll scan the folder for PDFs</p>
                </div>
            </div>
        )}

        <div className="p-5 border-b border-legal-800 flex items-center gap-2">
             <svg className="w-5 h-5 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
             </svg>
             <h1 className="text-lg font-serif font-bold tracking-wide">Legal AI</h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          
          {/* API Key Configuration - "Tune" */}
          <div className="mb-6 pb-6 border-b border-legal-800">
             <h2 className="text-xs font-semibold text-legal-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Configuration
             </h2>
             <div>
               <label className="text-xs text-legal-500 mb-1 block">Gemini API Key (Optional)</label>
               <input 
                  type="password" 
                  value={apiKey}
                  onChange={handleApiKeyChange}
                  placeholder="Enter Key or use Default..."
                  className="w-full bg-legal-800 border-legal-700 text-sm text-white rounded px-2 py-1 placeholder-legal-600 focus:outline-none focus:ring-1 focus:ring-yellow-500"
               />
               <p className="text-[10px] text-legal-500 mt-1">Leave empty to use the default system key.</p>
             </div>
          </div>

          {/* Projects Section */}
          <div className="mb-6 pb-6 border-b border-legal-800">
             <div className="flex items-center justify-between mb-4">
               <h2 className="text-xs font-semibold text-legal-400 uppercase tracking-wider flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                  </svg>
                  Projects ({projects.length}/{MAX_PROJECTS})
               </h2>
               <button
                 onClick={handleCreateProject}
                 disabled={projects.length >= MAX_PROJECTS}
                 className="text-xs bg-yellow-600 hover:bg-yellow-500 disabled:bg-legal-700 disabled:cursor-not-allowed text-white px-2 py-1 rounded transition-colors"
               >
                 + New
               </button>
             </div>
             
             <div className="space-y-1 max-h-48 overflow-y-auto">
               {projects.map(project => (
                 <div 
                   key={project.id}
                   className={`group flex items-center gap-2 p-2 rounded cursor-pointer transition-colors ${
                     currentProjectId === project.id 
                       ? 'bg-yellow-600 text-white' 
                       : 'bg-legal-800 hover:bg-legal-700 text-legal-300'
                   }`}
                 >
                   {editingProjectId === project.id ? (
                     <input
                       type="text"
                       value={editingName}
                       onChange={(e) => setEditingName(e.target.value)}
                       onBlur={() => handleRenameProject(project.id, editingName)}
                       onKeyDown={(e) => {
                         if (e.key === 'Enter') handleRenameProject(project.id, editingName);
                         if (e.key === 'Escape') setEditingProjectId(null);
                       }}
                       className="flex-1 bg-legal-900 text-white text-xs px-1 py-0.5 rounded focus:outline-none"
                       autoFocus
                     />
                   ) : (
                     <span 
                       className="flex-1 text-xs truncate"
                       onClick={() => handleSwitchProject(project.id)}
                       onDoubleClick={() => {
                         setEditingProjectId(project.id);
                         setEditingName(project.name);
                       }}
                     >
                       {project.name}
                     </span>
                   )}
                   
                   {/* Cross Memory Toggle */}
                   <button
                     onClick={(e) => {
                       e.stopPropagation();
                       handleToggleCrossMemory(project.id);
                     }}
                     title={project.crossMemory ? 'Cross-memory ON' : 'Cross-memory OFF'}
                     className={`p-1 rounded transition-colors ${
                       project.crossMemory 
                         ? 'text-green-400 hover:text-green-300' 
                         : 'text-legal-500 hover:text-legal-400'
                     }`}
                   >
                     <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                       <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                     </svg>
                   </button>
                   
                   {/* Delete button */}
                   {projects.length > 1 && (
                     <button
                       onClick={(e) => {
                         e.stopPropagation();
                         handleDeleteProject(project.id);
                       }}
                       className="opacity-0 group-hover:opacity-100 p-1 text-red-400 hover:text-red-300 transition-opacity"
                     >
                       <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                       </svg>
                     </button>
                   )}
                 </div>
               ))}
             </div>
             <p className="text-[10px] text-legal-500 mt-2">Double-click to rename. 🔗 = share memory across projects.</p>
          </div>

          <div className="mb-6">
            <h2 className="text-xs font-semibold text-legal-400 uppercase tracking-wider mb-4">Research Materials</h2>
            
            {/* Link Input */}
            <div className="mb-4">
               <label className="text-xs text-legal-500 mb-1 block">Add Web Reference (URL)</label>
               <div className="flex gap-2">
                 <input 
                    type="text" 
                    value={linkInput}
                    onChange={(e) => setLinkInput(e.target.value)}
                    placeholder="https://..."
                    className="flex-1 bg-legal-800 border-legal-700 text-sm text-white rounded px-2 py-1 placeholder-legal-600 focus:outline-none focus:ring-1 focus:ring-yellow-500"
                    onKeyDown={(e) => e.key === 'Enter' && handleAddLink()}
                 />
                 <button 
                    onClick={handleAddLink}
                    className="bg-legal-700 hover:bg-legal-600 text-white px-2 rounded"
                    title="Add URL Reference"
                 >
                    +
                 </button>
               </div>
            </div>

            {/* File Inputs */}
            <div className="mb-4 space-y-2">
               {/* Hidden Inputs */}
               <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="hidden"
                multiple
                accept=".pdf,.txt,.md,.csv"
              />
              <input
                type="file"
                ref={folderInputRef}
                onChange={handleFolderUpload}
                className="hidden"
                {...{ webkitdirectory: "", directory: "" } as any}
                multiple
              />

              <Button 
                variant="secondary" 
                className="w-full bg-legal-800 border-legal-700 text-legal-100 hover:bg-legal-700 hover:text-white"
                onClick={() => fileInputRef.current?.click()}
              >
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Upload Files
              </Button>
              
              <Button 
                variant="secondary" 
                className="w-full bg-legal-700 border-legal-700 text-legal-100 hover:bg-legal-600 hover:text-white"
                onClick={() => folderInputRef.current?.click()}
              >
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                Upload Additional Folder
              </Button>
              
              {/* Knowledge Base Status - Automatic */}
              {knowledgeBaseStatus.loaded && (
                <div className="bg-gradient-to-r from-green-900 to-legal-800 rounded-lg p-3 mt-3 border border-green-700">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full bg-green-400 animate-pulse"></div>
                    <span className="text-sm font-medium text-green-300">📚 Knowledge Base Active</span>
                  </div>
                </div>
              )}
              
              {!knowledgeBaseStatus.loaded && (
                <div className="bg-legal-800 rounded-md p-2 mt-2">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></div>
                    <span className="text-xs text-legal-300">Loading Knowledge Base...</span>
                  </div>
                </div>
              )}
            </div>
            
            <DocumentList documents={documents} onRemove={handleRemoveDocument} />
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-legal-950 border-t border-legal-800">
             <div className="flex items-center gap-3">
                 <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-yellow-400 to-yellow-600 flex items-center justify-center text-legal-900 font-bold text-xs">AI</div>
                 <div>
                     <p className="text-sm font-medium text-white">Gemini 3 Pro</p>
                     <p className="text-xs text-legal-400">
                        {apiKey ? 'Custom Key Active' : 'Default Key Active'}
                     </p>
                 </div>
             </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full relative z-10">
        <header className="bg-white border-b border-legal-200 h-14 flex items-center justify-between px-4 shadow-sm">
          <div className="flex items-center">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="mr-3 text-legal-500 hover:text-legal-800 focus:outline-none"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h2 className="text-base font-medium text-legal-900">Legal Research Workspace</h2>
          </div>
          <Button variant="ghost" onClick={handleReset} className="text-xs">
              Clear
          </Button>
        </header>

        <main className="flex-1 overflow-y-auto p-4 sm:p-6 bg-legal-50 scroll-smooth">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-xl mx-auto opacity-80">
              <svg className="w-16 h-16 text-legal-300 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              <h3 className="text-xl font-serif text-legal-800 mb-2">Legal AI</h3>
              <p className="text-sm text-legal-500 mb-2">
                  AI-powered legal research assistant
              </p>
              
              {/* Knowledge Base Status Banner */}
              {knowledgeBaseStatus.loaded && (
                <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-2 mb-4 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                  <span className="text-sm text-green-700 font-medium">
                    Knowledge Base Active
                  </span>
                </div>
              )}
              
              <p className="text-xs text-legal-400 mb-6">
                  Just ask your question
              </p>
              
              <div className="text-left w-full max-w-md bg-white p-4 rounded-md border border-legal-200">
                  <h4 className="text-xs font-bold text-legal-700 uppercase mb-2">Capabilities</h4>
                  <ul className="text-xs text-legal-600 space-y-1 list-disc pl-4">
                      <li>Essay</li>
                      <li>Problem questions</li>
                      <li>Advice</li>
                      <li>General legal questions</li>
                  </ul>
              </div>
              
              <div className="mt-4 text-left w-full max-w-md bg-yellow-50 p-3 rounded-md border border-yellow-200">
                  <h4 className="text-xs font-bold text-yellow-700 uppercase mb-1">💡 Try asking:</h4>
                  <ul className="text-xs text-yellow-600 space-y-0.5">
                      <li>"What are the key elements of a valid contract under English law?"</li>
                      <li>"Explain the duty of care in negligence under UK tort law"</li>
                      <li>"How does GDPR apply to AI systems processing personal data?"</li>
                  </ul>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto pb-8">
              {messages.map((msg) => (
                <ChatBubble 
                    key={msg.id} 
                    message={msg} 
                    onCitationClick={setActiveCitation} 
                />
              ))}
              <div ref={messagesEndRef} />
              
              {isLoading && (
                  <div className="flex justify-start mb-6">
                      <div className="bg-white border border-legal-200 rounded-lg rounded-bl-none shadow-sm px-4 py-3 flex items-center gap-3">
                          <div className="flex space-x-1">
                              <div className="w-1.5 h-1.5 bg-legal-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                              <div className="w-1.5 h-1.5 bg-legal-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                              <div className="w-1.5 h-1.5 bg-legal-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                          </div>
                          <span className="text-xs text-legal-500 font-medium">Researching & Reasoning...</span>
                      </div>
                  </div>
              )}
            </div>
          )}
        </main>

        <div className="p-4 bg-white border-t border-legal-200">
          <div className="max-w-3xl mx-auto relative">
             <div className="relative rounded-md shadow-sm">
                <textarea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                        }
                    }}
                    placeholder="Ask for an Essay, Case Analysis, or Client Advice..."
                    className="form-textarea block w-full rounded-lg border-legal-300 pl-4 pr-12 py-3 focus:border-legal-500 focus:ring-legal-500 sm:text-sm resize-none h-14"
                    disabled={isLoading}
                />
                <div className="absolute top-3 right-2">
                    <Button 
                        onClick={handleSendMessage} 
                        disabled={isLoading || !inputValue.trim()}
                        className="rounded-full w-8 h-8 p-0 flex items-center justify-center bg-legal-700 hover:bg-legal-900"
                    >
                         <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                         </svg>
                    </Button>
                </div>
             </div>
          </div>
        </div>
      </div>

      {/* Reference Viewer (Right Sidebar) */}
      <div className={`fixed inset-y-0 right-0 w-96 bg-white shadow-2xl transform transition-transform duration-300 ease-in-out border-l border-legal-200 z-30 flex flex-col ${activeCitation ? 'translate-x-0' : 'translate-x-full'}`}>
        <div className="p-5 border-b border-legal-200 bg-legal-50 flex items-center justify-between">
            <h3 className="text-sm font-bold text-legal-800 uppercase tracking-wide">Citation Source</h3>
            <button onClick={() => setActiveCitation(null)} className="text-legal-400 hover:text-legal-700">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
        
        {activeCitation && (
            <div className="p-6 flex-1 overflow-y-auto">
                <div className="mb-6">
                    <span className="text-xs font-semibold text-legal-500 uppercase block mb-1">Source</span>
                    <div className="flex items-center gap-2 text-legal-900 font-medium bg-gray-50 p-2 rounded border border-gray-200">
                         {activeCitation.doc.startsWith('http') ? (
                           <svg className="w-4 h-4 text-blue-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                           </svg>
                         ) : (
                           <svg className="w-4 h-4 text-legal-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                           </svg>
                         )}
                         <span className="truncate" title={activeCitation.doc}>{activeCitation.doc}</span>
                    </div>
                </div>

                <div className="mb-6">
                    <span className="text-xs font-semibold text-legal-500 uppercase block mb-1">Reference</span>
                    <p className="text-sm text-legal-800 font-bold font-serif">{activeCitation.ref}</p>
                    {activeCitation.loc && !shouldHideLocation(activeCitation.doc) && (
                        <span className="inline-block mt-1 text-xs bg-legal-100 text-legal-700 px-2 py-0.5 rounded">
                           Location: {activeCitation.loc}
                        </span>
                    )}
                </div>
            </div>
        )}
      </div>
    </div>
  );
};

export default App;