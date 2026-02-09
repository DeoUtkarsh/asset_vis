"""
RAG Ingestion Pipeline
======================
Builds vector store from Excel files for semantic search.

Run this once or whenever Excel data changes:
    python rag_ingest.py

This will create ./chroma_db/ folder with indexed data.
"""

import pandas as pd
import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ASSET_MODULE = Path("../Asset_Module")
CHROMA_DB_PATH = "./chroma_db"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    print("❌ ERROR: NVIDIA_API_KEY not found in environment variables!")
    print("   Create a .env file with: NVIDIA_API_KEY=your_key_here")
    exit(1)

# Initialize NVIDIA Embedding Model
embed_model = NVIDIAEmbedding(
    model="nvidia/nv-embedqa-e5-v5",
    api_key=NVIDIA_API_KEY,
    truncate="END"
)


def load_fmeca_documents():
    """
    Load FMECA Documents.xlsx and convert to searchable text chunks
    Each row becomes a document with full context
    """
    print("\n📚 Loading FMECA Documents...")
    
    fmeca_file = ASSET_MODULE / "FMECA Documents.xlsx"
    if not fmeca_file.exists():
        print(f"   ⚠️  FMECA file not found: {fmeca_file}")
        return []
    
    try:
        df = pd.read_excel(fmeca_file, sheet_name='FMECA')
        print(f"   Loaded {len(df)} FMECA records")
        
        docs = []
        for idx, row in df.iterrows():
            # Context injection: create rich text from structured data
            text_parts = []
            
            # Equipment and component context
            equipment = row.get('Equipment', 'Unknown')
            component = row.get('Component', 'Unknown')
            make = row.get('Make', '')
            
            text_parts.append(f"Equipment: {equipment}")
            if make:
                text_parts.append(f"Make: {make}")
            text_parts.append(f"Component: {component}")
            
            # Failure information
            failure_mode = row.get('Failure_Mode', '')
            failure_causes = row.get('Failure_Causes', '')
            failure_effects = row.get('Failure_Effects', '')
            failure_symptom = row.get('Failure_Symptom', '')
            
            if failure_mode:
                text_parts.append(f"Failure Mode: {failure_mode}")
            if failure_symptom:
                text_parts.append(f"Symptoms: {failure_symptom}")
            if failure_causes:
                text_parts.append(f"Causes: {failure_causes}")
            if failure_effects:
                text_parts.append(f"Effects: {failure_effects}")
            
            # Recommendations
            rec_checks = row.get('Recommended_Checks', '')
            rec_actions = row.get('Recommended_Actions', '')
            
            if rec_checks:
                text_parts.append(f"Recommended Checks: {rec_checks}")
            if rec_actions:
                text_parts.append(f"Recommended Actions: {rec_actions}")
            
            # Combine into single text block
            text = " | ".join(text_parts)
            
            # Create document with metadata
            docs.append(Document(
                text=text,
                metadata={
                    "source": "FMECA",
                    "equipment": str(equipment),
                    "component": str(component),
                    "failure_mode": str(failure_mode),
                    "row_id": row.get('Row_ID', f'fmeca_{idx}')
                }
            ))
        
        print(f"   ✅ Created {len(docs)} FMECA documents")
        return docs
    
    except Exception as e:
        print(f"   ❌ Error loading FMECA: {e}")
        return []


def load_alert_actions():
    """
    Load Alert actions.xlsx (all sheets) and convert to searchable chunks
    Each incident becomes a document with full context
    """
    print("\n🚨 Loading Alert Actions...")
    
    alert_file = ASSET_MODULE / "Alert actions.xlsx"
    if not alert_file.exists():
        print(f"   ⚠️  Alert actions file not found: {alert_file}")
        return []
    
    # Sheets to process
    sheets = [
        'Combination', 'JAN 2025', 'FEB 2025', 'MAR 2025', 'APR 2025',
        'MAY 2025', 'JUN 2025', 'JUL 2025', 'AUG 2025', 'SEP 2025',
        'OCT 2025', 'Alert Details'
    ]
    
    all_docs = []
    
    for sheet_name in sheets:
        try:
            df = pd.read_excel(alert_file, sheet_name=sheet_name)
            
            for idx, row in df.iterrows():
                # Skip rows with no vessel name
                vessel = row.get('Vessel Name', None)
                if pd.isna(vessel):
                    continue
                
                # Build text representation
                text_parts = []
                
                text_parts.append(f"Incident Record")
                text_parts.append(f"Vessel: {vessel}")
                
                # Date
                date_val = row.get('Date of occurence', '')
                if pd.notna(date_val):
                    text_parts.append(f"Date: {date_val}")
                
                # Fleet
                fleet = row.get('Fleet', '')
                if pd.notna(fleet):
                    text_parts.append(f"Fleet: {fleet}")
                
                # Alert category
                category = row.get('Category of alert', '')
                if pd.notna(category):
                    text_parts.append(f"Alert Type: {category}")
                
                # Description (CRITICAL - main content)
                description = row.get('Description of alert', '')
                if pd.notna(description):
                    text_parts.append(f"Description: {description}")
                
                # Action taken
                action = row.get('Action Taken', '')
                if pd.notna(action):
                    text_parts.append(f"Action Taken: {action}")
                
                # Remarks (CRITICAL - contains root cause and details)
                remarks = row.get('Remarks from TSI/Vessel', '')
                if pd.notna(remarks):
                    text_parts.append(f"Remarks: {remarks}")
                
                # Combine
                text = " | ".join(text_parts)
                
                # Create document
                all_docs.append(Document(
                    text=text,
                    metadata={
                        "source": "Incidents",
                        "vessel": str(vessel),
                        "sheet": sheet_name,
                        "category": str(category) if pd.notna(category) else "Unknown"
                    }
                ))
            
            print(f"   ✅ Processed {sheet_name}: {len(df)} incidents")
        
        except Exception as e:
            print(f"   ⚠️  Error processing {sheet_name}: {e}")
            continue
    
    print(f"   ✅ Created {len(all_docs)} incident documents")
    return all_docs


def load_dg_rma_parts():
    """
    Load DG RMA Analysis.xlsm and convert to searchable chunks
    Focus on parts consumption patterns and costs
    """
    print("\n🔩 Loading DG RMA Parts Data...")
    
    dg_file = ASSET_MODULE / "DG RMA Analysis.xlsm"
    if not dg_file.exists():
        print(f"   ⚠️  DG RMA file not found: {dg_file}")
        return []
    
    all_docs = []
    
    # Process Sheet2 and Sheet3 (similar structure)
    for sheet_name in ['Sheet2', 'Sheet3']:
        try:
            df = pd.read_excel(dg_file, sheet_name=sheet_name, header=2)
            
            for idx, row in df.iterrows():
                # Part name from SubComp
                part = row.get('SubComp', None)
                if pd.isna(part) or str(part).upper() in ['NA', 'N/A', 'NAN', '', 'SUBCOMP']:
                    continue
                
                # System and component
                system = row.get('SYS', 'Unknown')
                comp = row.get('Comp', 'Unknown')
                
                # Cost from man-hours
                mh = row.get('MH', 0)
                cost = 0
                if pd.notna(mh):
                    try:
                        cost = float(mh) * 50
                    except:
                        pass
                
                # Build text
                text = (
                    f"Spare Parts Record | "
                    f"System: {system} | "
                    f"Component: {comp} | "
                    f"Part Name: {part} | "
                    f"Estimated Cost: ${cost:.2f} | "
                    f"Source: {sheet_name}"
                )
                
                all_docs.append(Document(
                    text=text,
                    metadata={
                        "source": "Parts",
                        "part_name": str(part),
                        "system": str(system),
                        "sheet": sheet_name
                    }
                ))
            
            print(f"   ✅ Processed {sheet_name}: {len(df)} parts")
        
        except Exception as e:
            print(f"   ⚠️  Error processing {sheet_name}: {e}")
            continue
    
    # Process Sheet5 (wide format)
    try:
        df = pd.read_excel(dg_file, sheet_name='Sheet5')
        
        # Get part columns (exclude system columns)
        system_cols = ['SSDG1', 'SSDG2', 'SSDG3', 'SSDG4']
        part_cols = [col for col in df.columns 
                     if col not in system_cols 
                     and not str(col).startswith('Unnamed')
                     and len(str(col)) > 2]
        
        # For each part column, create a summary document
        for part_col in part_cols:
            try:
                total_consumption = df[part_col].sum()
                if pd.notna(total_consumption) and total_consumption > 0:
                    text = (
                        f"Spare Parts Summary | "
                        f"Part: {part_col} | "
                        f"Total Fleet Consumption: {int(total_consumption)} units | "
                        f"Used across multiple systems (SSDG1-4) | "
                        f"Source: Sheet5 wide format"
                    )
                    
                    all_docs.append(Document(
                        text=text,
                        metadata={
                            "source": "Parts",
                            "part_name": str(part_col),
                            "sheet": "Sheet5"
                        }
                    ))
            except:
                continue
        
        print(f"   ✅ Processed Sheet5: {len(part_cols)} part types")
    
    except Exception as e:
        print(f"   ⚠️  Error processing Sheet5: {e}")
    
    print(f"   ✅ Created {len(all_docs)} parts documents")
    return all_docs


def build_vector_store():
    """
    Main function: Load all data, embed, and store in ChromaDB
    """
    print("=" * 60)
    print("RAG VECTOR STORE BUILDER")
    print("=" * 60)
    
    # Load all documents
    fmeca_docs = load_fmeca_documents()
    incident_docs = load_alert_actions()
    parts_docs = load_dg_rma_parts()
    
    all_docs = fmeca_docs + incident_docs + parts_docs
    
    if len(all_docs) == 0:
        print("\n❌ ERROR: No documents loaded! Check Excel files.")
        return False
    
    print(f"\n📊 Total documents to index: {len(all_docs)}")
    print(f"   - FMECA: {len(fmeca_docs)}")
    print(f"   - Incidents: {len(incident_docs)}")
    print(f"   - Parts: {len(parts_docs)}")
    
    # Initialize ChromaDB
    print(f"\n💾 Initializing ChromaDB at: {CHROMA_DB_PATH}")
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Delete existing collection if present (fresh start)
    try:
        db.delete_collection("asset_knowledge")
        print("   🗑️  Deleted existing collection")
    except:
        pass
    
    chroma_collection = db.get_or_create_collection("asset_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index and embed documents
    print("\n🔄 Embedding documents (this may take 2-5 minutes)...")
    try:
        index = VectorStoreIndex.from_documents(
            all_docs,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        print("\n✅ SUCCESS! Vector store built and saved.")
        print(f"   Location: {Path(CHROMA_DB_PATH).absolute()}")
        print(f"   Documents indexed: {len(all_docs)}")
        
        # Save index metadata
        index.storage_context.persist(persist_dir=CHROMA_DB_PATH)
        
        return True
    
    except Exception as e:
        print(f"\n❌ ERROR during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = build_vector_store()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Vector store ready for chatbot!")
        print("   Next: Run chatbot_app.py to start the chat interface")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Build failed. Check errors above.")
        print("=" * 60)
        exit(1)
