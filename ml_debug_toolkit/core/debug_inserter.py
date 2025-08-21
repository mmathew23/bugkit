"""
Debug print inserter for automatic debugging code injection
"""

import ast
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseDebugTool


class DebugInserter(BaseDebugTool):
    """Automatically insert debug prints into Python code"""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        debug_prefix: str = "[DEBUG]",
        insert_at_function_start: bool = True,
        insert_at_function_end: bool = True,
        insert_before_returns: bool = True,
        insert_at_loops: bool = True,
        insert_at_conditionals: bool = False,
        track_variables: bool = True,
        track_tensor_shapes: bool = True,
        max_variable_length: int = 100,
    ):
        super().__init__(output_dir, verbose)
        self.debug_prefix = debug_prefix
        self.insert_at_function_start = insert_at_function_start
        self.insert_at_function_end = insert_at_function_end
        self.insert_before_returns = insert_before_returns
        self.insert_at_loops = insert_at_loops
        self.insert_at_conditionals = insert_at_conditionals
        self.track_variables = track_variables
        self.track_tensor_shapes = track_tensor_shapes
        self.max_variable_length = max_variable_length
        
        self.inserted_files: List[Path] = []
        self.insertion_stats: Dict[str, int] = {}
        
    def enable(self) -> None:
        """Enable debug inserter"""
        self.enabled = True
        if self.verbose:
            self.logger.info("Debug inserter enabled")
    
    def disable(self) -> None:
        """Disable debug inserter"""
        self.enabled = False
        if self.verbose:
            self.logger.info("Debug inserter disabled")
    
    def insert_debug_prints(
        self, 
        file_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        functions_to_instrument: Optional[List[str]] = None,
        exclude_functions: Optional[List[str]] = None
    ) -> Path:
        """Insert debug prints into a Python file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {file_path}: {e}")
            raise
        
        # Transform AST with debug prints
        transformer = DebugTransformer(
            debug_prefix=self.debug_prefix,
            insert_at_function_start=self.insert_at_function_start,
            insert_at_function_end=self.insert_at_function_end,
            insert_before_returns=self.insert_before_returns,
            insert_at_loops=self.insert_at_loops,
            insert_at_conditionals=self.insert_at_conditionals,
            track_variables=self.track_variables,
            track_tensor_shapes=self.track_tensor_shapes,
            max_variable_length=self.max_variable_length,
            functions_to_instrument=functions_to_instrument,
            exclude_functions=exclude_functions or []
        )
        
        transformed_tree = transformer.visit(tree)
        
        # Convert back to source code
        import astor
        transformed_code = astor.to_source(transformed_tree)
        
        # Add debug utilities import at the top
        debug_imports = self._generate_debug_imports()
        transformed_code = debug_imports + "\n\n" + transformed_code
        
        # Determine output path
        if output_path is None:
            output_path = self.output_dir / f"debug_{file_path.name}"
        else:
            output_path = Path(output_path)
        
        # Write transformed code
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transformed_code)
        
        # Update stats
        self.inserted_files.append(output_path)
        self.insertion_stats[str(file_path)] = transformer.insertion_count
        
        if self.verbose:
            self.logger.info(
                f"Inserted {transformer.insertion_count} debug prints into {file_path} "
                f"-> {output_path}"
            )
        
        return output_path
    
    def insert_debug_prints_directory(
        self,
        directory_path: Union[str, Path],
        output_directory: Optional[Union[str, Path]] = None,
        pattern: str = "*.py",
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Path]:
        """Insert debug prints into all Python files in a directory"""
        directory_path = Path(directory_path)
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", "test_*", "*_test.py"]
        
        if output_directory is None:
            output_directory = self.output_dir / "debug_code"
        else:
            output_directory = Path(output_directory)
        
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Find Python files
        search_pattern = "**/*.py" if recursive else "*.py"
        python_files = list(directory_path.glob(search_pattern))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in python_files:
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        # Process each file
        output_files = []
        for file_path in filtered_files:
            try:
                relative_path = file_path.relative_to(directory_path)
                output_path = output_directory / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                result_path = self.insert_debug_prints(file_path, output_path)
                output_files.append(result_path)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
        
        if self.verbose:
            self.logger.info(f"Processed {len(output_files)} files from {directory_path}")
        
        return output_files
    
    def _generate_debug_imports(self) -> str:
        """Generate necessary imports for debug functionality"""
        return '''
import sys
import time
import traceback
from typing import Any

def _debug_print(prefix: str, message: str, **kwargs):
    """Debug print utility"""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    frame = sys._getframe(1)
    location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    print(f"{prefix} [{timestamp}] {location} | {message}", **kwargs)

def _format_value(value: Any, max_length: int = 100) -> str:
    """Format value for debug printing"""
    try:
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            # Tensor-like object
            return f"{type(value).__name__}(shape={getattr(value, 'shape', 'unknown')}, dtype={getattr(value, 'dtype', 'unknown')})"
        elif hasattr(value, '__len__') and not isinstance(value, str):
            return f"{type(value).__name__}(len={len(value)})"
        else:
            str_repr = str(value)
            if len(str_repr) > max_length:
                return str_repr[:max_length] + "..."
            return str_repr
    except:
        return f"{type(value).__name__}(repr_failed)"

def _debug_vars(prefix: str, local_vars: dict, max_length: int = 100):
    """Print debug information about local variables"""
    filtered_vars = {k: v for k, v in local_vars.items() 
                    if not k.startswith('_') and k not in ['self', 'cls']}
    if filtered_vars:
        var_info = ", ".join([f"{k}={_format_value(v, max_length)}" for k, v in filtered_vars.items()])
        _debug_print(prefix, f"Variables: {var_info}")
        '''.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get insertion statistics"""
        return {
            "files_processed": len(self.inserted_files),
            "total_insertions": sum(self.insertion_stats.values()),
            "insertion_stats": self.insertion_stats,
            "output_files": [str(p) for p in self.inserted_files]
        }


class DebugTransformer(ast.NodeTransformer):
    """AST transformer to insert debug prints"""
    
    def __init__(
        self,
        debug_prefix: str = "[DEBUG]",
        insert_at_function_start: bool = True,
        insert_at_function_end: bool = True,
        insert_before_returns: bool = True,
        insert_at_loops: bool = True,
        insert_at_conditionals: bool = False,
        track_variables: bool = True,
        track_tensor_shapes: bool = True,
        max_variable_length: int = 100,
        functions_to_instrument: Optional[List[str]] = None,
        exclude_functions: Optional[List[str]] = None,
    ):
        self.debug_prefix = debug_prefix
        self.insert_at_function_start = insert_at_function_start
        self.insert_at_function_end = insert_at_function_end
        self.insert_before_returns = insert_before_returns
        self.insert_at_loops = insert_at_loops
        self.insert_at_conditionals = insert_at_conditionals
        self.track_variables = track_variables
        self.track_tensor_shapes = track_tensor_shapes
        self.max_variable_length = max_variable_length
        self.functions_to_instrument = functions_to_instrument
        self.exclude_functions = exclude_functions or []
        
        self.insertion_count = 0
        self.current_function = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform function definitions"""
        # Check if function should be instrumented
        if self.functions_to_instrument and node.name not in self.functions_to_instrument:
            return node
        if node.name in self.exclude_functions:
            return node
        
        self.current_function = node.name
        
        # Transform function body
        new_body = []
        
        # Insert debug print at function start
        if self.insert_at_function_start:
            args_str = ", ".join([arg.arg for arg in node.args.args])
            debug_call = self._create_debug_print(
                f"Entering function '{node.name}' with args: {args_str}"
            )
            new_body.append(debug_call)
            self.insertion_count += 1
            
            # Insert variable tracking if enabled
            if self.track_variables:
                var_debug_call = self._create_var_debug_print()
                new_body.append(var_debug_call)
                self.insertion_count += 1
        
        # Process existing body
        for stmt in node.body:
            if isinstance(stmt, ast.Return) and self.insert_before_returns:
                # Insert debug print before return
                return_debug = self._create_debug_print(
                    f"Returning from function '{node.name}'"
                )
                new_body.append(return_debug)
                self.insertion_count += 1
                
                # Track return value if it exists
                if stmt.value:
                    return_value_debug = self._create_debug_print(
                        f"Return value: {ast.unparse(stmt.value) if hasattr(ast, 'unparse') else 'value'}"
                    )
                    new_body.append(return_value_debug)
                    self.insertion_count += 1
            
            # Transform the statement recursively
            transformed_stmt = self.visit(stmt)
            if isinstance(transformed_stmt, list):
                new_body.extend(transformed_stmt)
            else:
                new_body.append(transformed_stmt)
        
        # Insert debug print at function end (for functions without explicit return)
        if (self.insert_at_function_end and 
            (not new_body or not isinstance(new_body[-1], ast.Return))):
            end_debug = self._create_debug_print(
                f"Exiting function '{node.name}'"
            )
            new_body.append(end_debug)
            self.insertion_count += 1
        
        node.body = new_body
        self.current_function = None
        return node
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Transform for loops"""
        if self.insert_at_loops:
            # Insert debug print at loop start
            loop_debug = self._create_debug_print(
                f"Starting for loop: {ast.unparse(node.target) if hasattr(ast, 'unparse') else 'loop'}"
            )
            
            # Transform loop body
            new_body = [loop_debug]
            self.insertion_count += 1
            
            for stmt in node.body:
                transformed_stmt = self.visit(stmt)
                if isinstance(transformed_stmt, list):
                    new_body.extend(transformed_stmt)
                else:
                    new_body.append(transformed_stmt)
            
            node.body = new_body
        else:
            node = self.generic_visit(node)
        
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Transform while loops"""
        if self.insert_at_loops:
            # Insert debug print at loop start
            loop_debug = self._create_debug_print("Starting while loop")
            
            # Transform loop body
            new_body = [loop_debug]
            self.insertion_count += 1
            
            for stmt in node.body:
                transformed_stmt = self.visit(stmt)
                if isinstance(transformed_stmt, list):
                    new_body.extend(transformed_stmt)
                else:
                    new_body.append(transformed_stmt)
            
            node.body = new_body
        else:
            node = self.generic_visit(node)
        
        return node
    
    def visit_If(self, node: ast.If) -> ast.If:
        """Transform if statements"""
        if self.insert_at_conditionals:
            # Insert debug print for if condition
            if_debug = self._create_debug_print("Entering if statement")
            
            new_body = [if_debug]
            self.insertion_count += 1
            
            for stmt in node.body:
                transformed_stmt = self.visit(stmt)
                if isinstance(transformed_stmt, list):
                    new_body.extend(transformed_stmt)
                else:
                    new_body.append(transformed_stmt)
            
            node.body = new_body
            
            # Transform else body if it exists
            if node.orelse:
                else_debug = self._create_debug_print("Entering else statement")
                new_orelse = [else_debug]
                self.insertion_count += 1
                
                for stmt in node.orelse:
                    transformed_stmt = self.visit(stmt)
                    if isinstance(transformed_stmt, list):
                        new_orelse.extend(transformed_stmt)
                    else:
                        new_orelse.append(transformed_stmt)
                
                node.orelse = new_orelse
        else:
            node = self.generic_visit(node)
        
        return node
    
    def _create_debug_print(self, message: str) -> ast.Expr:
        """Create a debug print statement"""
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='_debug_print', ctx=ast.Load()),
                args=[
                    ast.Constant(value=self.debug_prefix),
                    ast.Constant(value=message)
                ],
                keywords=[]
            )
        )
    
    def _create_var_debug_print(self) -> ast.Expr:
        """Create a variable debug print statement"""
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='_debug_vars', ctx=ast.Load()),
                args=[
                    ast.Constant(value=self.debug_prefix),
                    ast.Call(
                        func=ast.Name(id='locals', ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    ),
                    ast.Constant(value=self.max_variable_length)
                ],
                keywords=[]
            )
        )