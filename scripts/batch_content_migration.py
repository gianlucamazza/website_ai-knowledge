#!/usr/bin/env python3
"""
Batch Content Migration Tool
Enterprise-grade tool for migrating existing markdown content to new quality standards

Handles:
- Safe batch processing with rollback capabilities
- Performance optimization for large content sets (100+ files)
- Detailed migration reports and analytics
- Validation of migration results
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass
import hashlib

# Import our quality fixer
sys.path.append(str(Path(__file__).parent))
from markdown_quality_fixer import MarkdownQualityFixer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationResult:
    """Result of migrating a single file"""
    file_path: str
    original_hash: str
    new_hash: str
    fixes_applied: Dict[str, int]
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0

@dataclass  
class MigrationSummary:
    """Summary of entire migration operation"""
    total_files: int
    successful_migrations: int
    failed_migrations: int
    files_unchanged: int
    total_fixes: Dict[str, int]
    processing_time: float
    rollback_available: bool

class BatchContentMigrator:
    """Enterprise batch content migration with safety features"""
    
    def __init__(self, 
                 content_dir: str,
                 backup_dir: Optional[str] = None,
                 max_workers: int = 4,
                 chunk_size: int = 10):
        self.content_dir = Path(content_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.content_dir.parent / 'content_backup'
        self.max_workers = min(max_workers, os.cpu_count() or 1)
        self.chunk_size = chunk_size
        
        # Migration state
        self.migration_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.migration_log_dir = Path('migration_logs') 
        self.migration_log_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results: List[MigrationResult] = []
        self.failed_files: Set[str] = set()
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
            
    def create_full_backup(self) -> bool:
        """Create full backup of content directory"""
        try:
            backup_path = self.backup_dir / f'backup_{self.migration_id}'
            
            if backup_path.exists():
                shutil.rmtree(backup_path)
                
            logger.info(f"Creating full backup at: {backup_path}")
            shutil.copytree(self.content_dir, backup_path)
            
            # Create backup manifest
            manifest_path = backup_path / 'backup_manifest.json'
            manifest = {
                'migration_id': self.migration_id,
                'created_at': datetime.now().isoformat(),
                'source_dir': str(self.content_dir),
                'file_count': len(list(backup_path.rglob('*.md')))
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            logger.info(f"Backup created successfully: {manifest['file_count']} files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
            
    def discover_content_files(self) -> List[Path]:
        """Discover all markdown files to migrate"""
        patterns = ['**/*.md']
        files = []
        
        for pattern in patterns:
            files.extend(self.content_dir.rglob(pattern))
            
        # Filter out certain files
        exclude_patterns = [
            '**/node_modules/**',
            '**/dist/**', 
            '**/coverage/**',
            '**/.git/**',
            '**/test-results/**',
            '**/*.backup',
        ]
        
        filtered_files = []
        for file_path in files:
            exclude = False
            for exclude_pattern in exclude_patterns:
                if file_path.match(exclude_pattern):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(file_path)
                
        logger.info(f"Discovered {len(filtered_files)} markdown files for migration")
        return filtered_files
        
    def migrate_file(self, file_path: Path) -> MigrationResult:
        """Migrate a single file with detailed tracking"""
        start_time = datetime.now()
        
        try:
            # Calculate original hash
            original_hash = self.calculate_file_hash(file_path)
            
            # Create file-specific fixer instance
            fixer = MarkdownQualityFixer(
                content_dir=str(file_path.parent),
                dry_run=False,
                backup=False  # We handle backups at batch level
            )
            
            # Process the single file
            success = fixer.fix_markdown_file(file_path)
            
            # Calculate new hash
            new_hash = self.calculate_file_hash(file_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MigrationResult(
                file_path=str(file_path),
                original_hash=original_hash,
                new_hash=new_hash,
                fixes_applied=fixer.fixes_applied.copy(),
                success=success,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to migrate {file_path}: {e}")
            
            return MigrationResult(
                file_path=str(file_path),
                original_hash="",
                new_hash="",
                fixes_applied={},
                success=False,
                error=str(e),
                processing_time=processing_time
            )
            
    def migrate_batch(self, files: List[Path]) -> List[MigrationResult]:
        """Migrate a batch of files in parallel"""
        results = []
        
        logger.info(f"Processing batch of {len(files)} files...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.migrate_file, file_path): file_path 
                for file_path in files
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        logger.debug(f"✅ Migrated: {file_path}")
                    else:
                        logger.error(f"❌ Failed: {file_path} - {result.error}")
                        self.failed_files.add(str(file_path))
                        
                except Exception as e:
                    logger.error(f"Exception processing {file_path}: {e}")
                    results.append(MigrationResult(
                        file_path=str(file_path),
                        original_hash="",
                        new_hash="",
                        fixes_applied={},
                        success=False,
                        error=f"Exception: {e}"
                    ))
                    
        return results
        
    def validate_migration_results(self) -> bool:
        """Validate that migration results are consistent"""
        logger.info("Validating migration results...")
        
        validation_errors = []
        
        for result in self.results:
            file_path = Path(result.file_path)
            
            # Check file still exists
            if not file_path.exists():
                validation_errors.append(f"File missing after migration: {file_path}")
                continue
                
            # Check file is readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) == 0:
                        validation_errors.append(f"File is empty after migration: {file_path}")
            except Exception as e:
                validation_errors.append(f"Cannot read file after migration: {file_path} - {e}")
                
        if validation_errors:
            logger.error(f"Validation failed with {len(validation_errors)} errors:")
            for error in validation_errors[:10]:  # Show first 10
                logger.error(f"  {error}")
            if len(validation_errors) > 10:
                logger.error(f"  ... and {len(validation_errors) - 10} more errors")
            return False
            
        logger.info("✅ Migration validation passed")
        return True
        
    def generate_migration_report(self) -> MigrationSummary:
        """Generate comprehensive migration report"""
        # Calculate totals
        total_files = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)
        unchanged = sum(1 for r in self.results if r.original_hash == r.new_hash)
        
        # Aggregate fixes
        total_fixes = {}
        for result in self.results:
            for fix_type, count in result.fixes_applied.items():
                total_fixes[fix_type] = total_fixes.get(fix_type, 0) + count
                
        # Calculate total processing time
        total_time = sum(r.processing_time for r in self.results)
        
        return MigrationSummary(
            total_files=total_files,
            successful_migrations=successful,
            failed_migrations=failed,
            files_unchanged=unchanged,
            total_fixes=total_fixes,
            processing_time=total_time,
            rollback_available=self.backup_dir.exists()
        )
        
    def save_detailed_report(self, summary: MigrationSummary) -> Path:
        """Save detailed migration report to disk"""
        report_path = self.migration_log_dir / f'migration_report_{self.migration_id}.json'
        
        report_data = {
            'migration_id': self.migration_id,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': summary.total_files,
                'successful_migrations': summary.successful_migrations,
                'failed_migrations': summary.failed_migrations,
                'files_unchanged': summary.files_unchanged,
                'total_fixes': summary.total_fixes,
                'processing_time': summary.processing_time,
                'rollback_available': summary.rollback_available,
            },
            'file_results': [
                {
                    'file_path': r.file_path,
                    'success': r.success,
                    'fixes_applied': r.fixes_applied,
                    'processing_time': r.processing_time,
                    'changed': r.original_hash != r.new_hash,
                    'error': r.error
                }
                for r in self.results
            ],
            'failed_files': list(self.failed_files)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Detailed report saved: {report_path}")
        return report_path
        
    def rollback_migration(self, migration_id: Optional[str] = None) -> bool:
        """Rollback to previous backup"""
        rollback_id = migration_id or self.migration_id
        backup_path = self.backup_dir / f'backup_{rollback_id}'
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
            
        try:
            logger.info(f"Rolling back to backup: {backup_path}")
            
            # Remove current content
            if self.content_dir.exists():
                shutil.rmtree(self.content_dir)
                
            # Restore from backup
            shutil.copytree(backup_path, self.content_dir)
            
            logger.info("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
            
    def run_migration(self, 
                     dry_run: bool = False,
                     create_backup: bool = True) -> MigrationSummary:
        """Run the complete migration process"""
        start_time = datetime.now()
        
        logger.info(f"Starting batch content migration {'(DRY RUN)' if dry_run else ''}")
        logger.info(f"Migration ID: {self.migration_id}")
        logger.info(f"Content directory: {self.content_dir}")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info(f"Chunk size: {self.chunk_size}")
        
        # Discover files
        files = self.discover_content_files()
        if not files:
            logger.warning("No files found to migrate")
            return MigrationSummary(0, 0, 0, 0, {}, 0.0, False)
            
        # Create backup if requested
        if create_backup and not dry_run:
            if not self.create_full_backup():
                logger.error("Failed to create backup - aborting migration")
                sys.exit(1)
                
        # Process files in chunks for better memory management
        if dry_run:
            logger.info("DRY RUN: Would process the following files:")
            for file_path in files[:10]:  # Show first 10
                logger.info(f"  {file_path}")
            if len(files) > 10:
                logger.info(f"  ... and {len(files) - 10} more files")
                
            return MigrationSummary(len(files), 0, 0, len(files), {}, 0.0, create_backup)
            
        # Process in chunks
        for i in range(0, len(files), self.chunk_size):
            chunk = files[i:i + self.chunk_size]
            logger.info(f"Processing chunk {i//self.chunk_size + 1}/{(len(files)-1)//self.chunk_size + 1}")
            
            chunk_results = self.migrate_batch(chunk)
            self.results.extend(chunk_results)
            
        # Validate results
        if not self.validate_migration_results():
            logger.error("Migration validation failed - consider rollback")
            
        # Generate and save report
        summary = self.generate_migration_report()
        summary.processing_time = (datetime.now() - start_time).total_seconds()
        
        report_path = self.save_detailed_report(summary)
        
        # Log summary
        logger.info("🎯 Migration Summary:")
        logger.info(f"  Total files: {summary.total_files}")
        logger.info(f"  Successful: {summary.successful_migrations}")
        logger.info(f"  Failed: {summary.failed_migrations}")
        logger.info(f"  Unchanged: {summary.files_unchanged}")
        logger.info(f"  Processing time: {summary.processing_time:.2f}s")
        
        if summary.total_fixes:
            logger.info("  Fixes applied:")
            for fix_type, count in summary.total_fixes.items():
                logger.info(f"    {fix_type}: {count}")
                
        if summary.failed_migrations > 0:
            logger.warning(f"⚠️  {summary.failed_migrations} files failed migration")
            logger.warning(f"Check detailed report: {report_path}")
            
        if summary.rollback_available:
            logger.info(f"💾 Rollback available: python {__file__} --rollback {self.migration_id}")
            
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch Content Migration Tool for AI Knowledge Website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be migrated
  python batch_content_migration.py apps/site/src/content --dry-run
  
  # Full migration with backup
  python batch_content_migration.py apps/site/src/content --backup-dir ./backups
  
  # High-performance migration
  python batch_content_migration.py apps/site/src/content --max-workers 8 --chunk-size 20
  
  # Rollback migration
  python batch_content_migration.py apps/site/src/content --rollback 20231205_143022
        """
    )
    
    parser.add_argument(
        'content_dir',
        help='Directory containing markdown content to migrate'
    )
    parser.add_argument(
        '--backup-dir',
        help='Directory for backups (default: content_dir/../content_backup)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup (dangerous!)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10,
        help='Number of files to process per chunk (default: 10)'
    )
    parser.add_argument(
        '--rollback',
        help='Rollback to specified migration ID'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize migrator
    migrator = BatchContentMigrator(
        content_dir=args.content_dir,
        backup_dir=args.backup_dir,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size
    )
    
    try:
        # Handle rollback
        if args.rollback:
            if migrator.rollback_migration(args.rollback):
                logger.info("Rollback completed successfully")
                sys.exit(0)
            else:
                logger.error("Rollback failed")
                sys.exit(1)
                
        # Run migration
        summary = migrator.run_migration(
            dry_run=args.dry_run,
            create_backup=not args.no_backup
        )
        
        # Exit with appropriate code
        if summary.failed_migrations > 0:
            logger.error(f"Migration completed with {summary.failed_migrations} failures")
            sys.exit(1)
        else:
            logger.info("Migration completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()