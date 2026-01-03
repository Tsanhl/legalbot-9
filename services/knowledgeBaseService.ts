/**
 * Knowledge Base Service
 * Manages the Law Resources as a default knowledge base for the AI
 */

import { UploadedDocument } from '../types';

export interface LawResourceEntry {
  id: string;
  name: string;
  path: string;
  category: string;
  subcategory: string;
  mimeType: string;
  size: number;
}

export interface LawResourceIndex {
  generatedAt: string;
  totalFiles: number;
  categories: string[];
  resources: LawResourceEntry[];
}

// Will be populated from the index file
let lawResourceIndex: LawResourceIndex | null = null;
let loadedResources: Map<string, UploadedDocument> = new Map();

// Cache for loaded PDF data
const pdfCache: Map<string, string> = new Map();

/**
 * Load the law resources index from the JSON file
 */
export async function loadLawResourceIndex(): Promise<LawResourceIndex | null> {
  if (lawResourceIndex) return lawResourceIndex;
  
  try {
    const response = await fetch('/law-resources-index.json');
    if (response.ok) {
      lawResourceIndex = await response.json();
      console.log(`📚 Loaded ${lawResourceIndex?.totalFiles} law resources from index`);
      return lawResourceIndex;
    }
  } catch (error) {
    console.warn('Could not load law resources index:', error);
  }
  return null;
}

/**
 * Get all available categories
 */
export function getCategories(): string[] {
  return lawResourceIndex?.categories || [];
}

/**
 * Get resources by category
 */
export function getResourcesByCategory(category: string): LawResourceEntry[] {
  if (!lawResourceIndex) return [];
  return lawResourceIndex.resources.filter(r => r.category === category);
}

/**
 * Search resources by name
 */
export function searchResources(query: string): LawResourceEntry[] {
  if (!lawResourceIndex) return [];
  const lowerQuery = query.toLowerCase();
  return lawResourceIndex.resources.filter(r => 
    r.name.toLowerCase().includes(lowerQuery) ||
    r.category.toLowerCase().includes(lowerQuery)
  );
}

/**
 * Load a specific PDF from the law resources folder
 * Returns base64 encoded data
 */
export async function loadResourcePdf(resource: LawResourceEntry): Promise<UploadedDocument | null> {
  // Check cache first
  if (loadedResources.has(resource.id)) {
    return loadedResources.get(resource.id)!;
  }
  
  try {
    const response = await fetch(`/${encodeURIComponent(resource.path)}`);
    if (!response.ok) {
      throw new Error(`Failed to load: ${resource.path}`);
    }
    
    const blob = await response.blob();
    const base64 = await blobToBase64(blob);
    
    const doc: UploadedDocument = {
      id: resource.id,
      type: 'file',
      name: resource.name + '.pdf',
      mimeType: 'application/pdf',
      data: base64,
      size: resource.size
    };
    
    loadedResources.set(resource.id, doc);
    return doc;
  } catch (error) {
    console.error(`Failed to load PDF: ${resource.path}`, error);
    return null;
  }
}

/**
 * Load multiple resources by IDs
 */
export async function loadResources(resourceIds: string[]): Promise<UploadedDocument[]> {
  if (!lawResourceIndex) return [];
  
  const docs: UploadedDocument[] = [];
  
  for (const id of resourceIds) {
    const resource = lawResourceIndex.resources.find(r => r.id === id);
    if (resource) {
      const doc = await loadResourcePdf(resource);
      if (doc) docs.push(doc);
    }
  }
  
  return docs;
}

/**
 * Helper to convert Blob to Base64
 */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result.split(',')[1]); // Remove data URL prefix
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Get a comprehensive summary of the knowledge base for the AI system prompt
 * This provides the AI with detailed information about available resources
 */
export function getKnowledgeBaseSummary(): string {
  if (!lawResourceIndex) {
    return 'No knowledge base loaded.';
  }
  
  // Build detailed category information with sample documents
  const categoryDetails = lawResourceIndex.categories.map(cat => {
    const resources = lawResourceIndex!.resources.filter(r => r.category === cat);
    const sampleDocs = resources.slice(0, 5).map(r => `    • ${r.name}`).join('\n');
    return `
📁 ${cat.toUpperCase()} (${resources.length} documents)
${sampleDocs}${resources.length > 5 ? `\n    ... and ${resources.length - 5} more documents` : ''}`;
  }).join('\n');
  
  return `
================================================================================
DEFAULT KNOWLEDGE BASE: UK LAW RESOURCES LIBRARY
================================================================================

Total Documents Available: ${lawResourceIndex.totalFiles} legal texts, cases, and academic materials

AVAILABLE CATEGORIES:
${categoryDetails}

================================================================================
USAGE INSTRUCTIONS (MANDATORY):
================================================================================

1. PRIMARY SOURCE RULE: For ANY legal question, FIRST check if relevant materials exist in this knowledge base. Use these as your primary authoritative sources.

2. CATEGORY MAPPING - When a user asks about:
   • Contract law → Use "Contract law" category documents
   • Torts, negligence, duty of care → Use "Tort law" category documents  
   • Trusts, fiduciary duties, trustees → Use "Trusts law" category documents
   • Pensions, pension schemes, trustees → Use "Pensions Law" category documents
   • Criminal offences, criminal liability → Use "Criminal law" category documents
   • EU law, European Union, Brexit → Use "EU law" category documents
   • Competition, antitrust, monopoly → Use "Competition Law" category documents
   • Commercial transactions, sale of goods → Use "Commercial Law" or "Commercial law revision" documents
   • Business, company law, employment → Use "Business law" category documents
   • AI, data protection, GDPR, privacy → Use "Ai and data protection act" category documents
   • Bioethics, medical law → Use "Biolaw" category documents
   • Mediation, ADR, dispute resolution → Use "International Commercial Mediation" category documents

3. CITATION RULE: When referencing knowledge from this database, cite the document name using OSCOLA format. Example:
   [[{"ref": "Caparo Industries plc v Dickman [1990] UKHL 2", "doc": "Tort law/Caparo case.pdf", "loc": ""}]]

4. OSCOLA GUIDE: The knowledge base includes the official OSCOLA 4th Edition referencing guide - use this for all citation formatting.

5. CROSS-REFERENCE: For complex questions spanning multiple areas (e.g., "AI in employment discrimination"), draw from multiple relevant categories.

YOU ARE NOW A LEGAL RESEARCH ASSISTANT WITH ACCESS TO THIS COMPREHENSIVE UK LAW LIBRARY.
Answer all legal questions using this knowledge base as your primary reference.
`;
}

/**
 * Get relevant resources for a query (simple keyword matching)
 * This can be enhanced with semantic search later
 */
export function getRelevantResources(query: string, limit: number = 10): LawResourceEntry[] {
  if (!lawResourceIndex) return [];
  
  const lowerQuery = query.toLowerCase();
  const keywords = lowerQuery.split(/\s+/).filter(k => k.length > 3);
  
  // Score each resource based on keyword matches
  const scored = lawResourceIndex.resources.map(resource => {
    let score = 0;
    const resourceText = `${resource.name} ${resource.category} ${resource.subcategory}`.toLowerCase();
    
    for (const keyword of keywords) {
      if (resourceText.includes(keyword)) {
        score += 1;
        // Bonus for exact word match
        if (resourceText.split(/\s+/).includes(keyword)) {
          score += 2;
        }
      }
    }
    
    return { resource, score };
  });
  
  // Sort by score and return top matches
  return scored
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(s => s.resource);
}

export { lawResourceIndex };

