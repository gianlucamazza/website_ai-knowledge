#!/usr/bin/env python3
"""
Fix flake8 issues automatically in the pipelines codebase.
"""

import os
import re
import subprocess
from pathlib import Path


def remove_unused_imports(file_path: Path, unused_imports: list):
    """Remove unused imports from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        should_keep = True
        for unused in unused_imports:
            # Handle different import patterns
            patterns = [
                f"^import {re.escape(unused)}$",
                f"^from .+ import .* {re.escape(unused)}",
                f"^from .+ import {re.escape(unused)}$",
                f"import .* {re.escape(unused)} ",
                f"import {re.escape(unused)},",
                f", {re.escape(unused)}$",
                f", {re.escape(unused)},",
            ]
            
            for pattern in patterns:
                if re.search(pattern, line.strip()):
                    # Check if it's the only import on the line
                    if "import" in line and unused in line:
                        # Remove just this import if multiple
                        new_line = line
                        new_line = re.sub(f", {re.escape(unused)}", "", new_line)
                        new_line = re.sub(f"{re.escape(unused)}, ", "", new_line)
                        new_line = re.sub(f"{re.escape(unused)}", "", new_line)
                        
                        # Clean up empty imports
                        if "from" in new_line and "import" in new_line:
                            parts = new_line.split("import")
                            if len(parts) > 1 and not parts[1].strip():
                                should_keep = False
                                modified = True
                                break
                            elif parts[1].strip() != line.split("import")[1].strip():
                                line = new_line
                                modified = True
        
        if should_keep:
            new_lines.append(line)
    
    # Clean up multiple blank lines
    cleaned_lines = []
    prev_blank = False
    for line in new_lines:
        if line.strip() == "":
            if not prev_blank:
                cleaned_lines.append(line)
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(cleaned_lines)
        return True
    return False


def fix_line_length(file_path: Path, long_lines: dict):
    """Fix lines that are too long."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    for line_num, line_content in long_lines.items():
        idx = line_num - 1
        if idx < len(lines):
            line = lines[idx]
            # Try to break at logical points
            if len(line) > 100:
                # For comments
                if "#" in line:
                    lines[idx] = line[:95] + "  # ...\n"
                # For strings
                elif '"' in line or "'" in line:
                    # Try to break at commas or spaces
                    if "," in line[80:]:
                        break_point = line.rfind(",", 80, 100) + 1
                        lines[idx] = line[:break_point] + "\n" + " " * 8 + line[break_point:]
                    else:
                        lines[idx] = line[:95] + '"\n' + ' " ' + line[95:]
                modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return True
    return False


def fix_comparison_to_true(file_path: Path, comparison_lines: list):
    """Fix comparisons to True/False."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    modified = False
    for line_num in comparison_lines:
        # Replace "== True" with just the condition
        content = re.sub(r"(\w+)\s*==\s*True", r"\1", content)
        content = re.sub(r"(\w+)\s*==\s*False", r"not \1", content)
        modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False


def fix_whitespace_issues(file_path: Path):
    """Fix trailing whitespace and blank lines with whitespace."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    for line in lines:
        # Remove trailing whitespace
        new_line = line.rstrip() + '\n' if line.strip() else '\n'
        if new_line != line:
            modified = True
        new_lines.append(new_line)
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        return True
    return False


def main():
    """Main function to fix all flake8 issues."""
    # Run flake8 and capture output
    result = subprocess.run(
        ["python", "-m", "flake8", "pipelines/", "--max-line-length=100", "--extend-ignore=E203,W503"],
        capture_output=True, text=True
    )
    
    issues = result.stdout.strip().split('\n')
    
    # Parse issues by file
    file_issues = {}
    for issue in issues:
        if issue and ':' in issue:
            parts = issue.split(':')
            if len(parts) >= 4:
                file_path = parts[0]
                line_num = int(parts[1])
                col_num = int(parts[2])
                error_code = parts[3].split()[0]
                error_msg = ':'.join(parts[3:]).strip()
                
                if file_path not in file_issues:
                    file_issues[file_path] = {
                        'unused_imports': [],
                        'long_lines': {},
                        'comparisons': [],
                        'unused_vars': {},
                        'other': []
                    }
                
                # Categorize issues
                if error_code == 'F401':  # unused import
                    # Extract the unused import name
                    match = re.search(r"'([^']+)' imported but unused", error_msg)
                    if match:
                        import_name = match.group(1)
                        # Get just the last part for things like typing.Dict
                        if '.' in import_name:
                            import_name = import_name.split('.')[-1]
                        file_issues[file_path]['unused_imports'].append(import_name)
                
                elif error_code == 'E501':  # line too long
                    file_issues[file_path]['long_lines'][line_num] = error_msg
                
                elif error_code == 'E712':  # comparison to True/False
                    file_issues[file_path]['comparisons'].append(line_num)
                
                elif error_code == 'F841':  # unused variable
                    match = re.search(r"local variable '([^']+)' is assigned", error_msg)
                    if match:
                        file_issues[file_path]['unused_vars'][line_num] = match.group(1)
                
                elif error_code in ['W293', 'W291']:  # whitespace issues
                    # Will handle these globally
                    pass
                
                else:
                    file_issues[file_path]['other'].append((line_num, error_code, error_msg))
    
    # Fix issues file by file
    total_fixed = 0
    for file_path, issues in file_issues.items():
        file_path = Path(file_path)
        if not file_path.exists():
            continue
        
        print(f"Fixing {file_path}...")
        
        # Fix unused imports
        if issues['unused_imports']:
            if remove_unused_imports(file_path, issues['unused_imports']):
                total_fixed += len(issues['unused_imports'])
                print(f"  - Removed {len(issues['unused_imports'])} unused imports")
        
        # Fix line length
        if issues['long_lines']:
            if fix_line_length(file_path, issues['long_lines']):
                total_fixed += len(issues['long_lines'])
                print(f"  - Fixed {len(issues['long_lines'])} long lines")
        
        # Fix comparisons
        if issues['comparisons']:
            if fix_comparison_to_true(file_path, issues['comparisons']):
                total_fixed += len(issues['comparisons'])
                print(f"  - Fixed {len(issues['comparisons'])} comparisons")
        
        # Fix whitespace
        if fix_whitespace_issues(file_path):
            print(f"  - Fixed whitespace issues")
        
        # Handle special cases
        if str(file_path).endswith('run_graph.py'):
            # Fix undefined ContentStatus
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Add ContentStatus import where needed
            if "ContentStatus.published" in content and "from .database.models import ContentStatus" not in content:
                # Find the right place to add import
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "from .database.models import" in line and "ContentStatus" not in line:
                        lines[i] = line.replace("Article", "Article, ContentStatus")
                        break
                
                with open(file_path, 'w') as f:
                    f.write('\n'.join(lines))
                print(f"  - Added missing ContentStatus import")
        
        # Fix f-string without placeholders
        if str(file_path).endswith('summarizer.py'):
            with open(file_path, 'r') as f:
                content = f.read()
            content = re.sub(r'f"([^{}"]*)"', r'"\1"', content)
            content = re.sub(r"f'([^{}']*)'", r"'\1'", content)
            with open(file_path, 'w') as f:
                f.write(content)
    
    print(f"\nTotal issues fixed: {total_fixed}")
    
    # Run flake8 again to verify
    result = subprocess.run(
        ["python", "-m", "flake8", "pipelines/", "--max-line-length=100", "--extend-ignore=E203,W503"],
        capture_output=True, text=True
    )
    
    remaining = len([l for l in result.stdout.strip().split('\n') if l])
    if remaining > 0:
        print(f"Remaining issues: {remaining}")
        print("\nRemaining issues that need manual fix:")
        print(result.stdout[:500])
    else:
        print("All flake8 issues fixed!")


if __name__ == "__main__":
    main()