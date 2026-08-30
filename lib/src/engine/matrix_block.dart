import 'dart:typed_data';

/// A row-major rectangular view onto a [Float64List].
///
/// The engine keeps the full transition and process-noise matrices in a single
/// flat buffer and hands each component a view of its own diagonal block. The
/// [stride] is the row length of the *underlying* matrix, which is why a block
/// can be written without knowing where it sits or copying anything afterwards.
///
/// Indices are not bounds-checked beyond what the typed list does itself; this
/// type is internal and lives in the filter's hot loop.
class MatrixBlock {
  /// A view of [rows] x [cols] starting at [offset], where consecutive rows of
  /// the view are [stride] elements apart in [storage].
  MatrixBlock(this.storage, this.offset, this.stride, this.rows, this.cols);

  /// A standalone [rows] x [cols] matrix with its own backing store.
  MatrixBlock.dense(int rows, int cols)
      : this(Float64List(rows * cols), 0, cols, rows, cols);

  /// The buffer the view reads and writes.
  final Float64List storage;

  /// Index in [storage] of this view's top-left entry.
  final int offset;

  /// Distance in [storage] between consecutive rows of the underlying matrix.
  final int stride;

  /// Rows in the view.
  final int rows;

  /// Columns in the view.
  final int cols;

  /// Entry `(i, j)`.
  double at(int i, int j) => storage[offset + i * stride + j];

  /// Sets entry `(i, j)`.
  void set(int i, int j, double value) {
    storage[offset + i * stride + j] = value;
  }

  /// Adds [value] to entry `(i, j)`.
  void add(int i, int j, double value) {
    storage[offset + i * stride + j] += value;
  }

  /// Writes [value] into every entry of the view.
  void fill(double value) {
    for (var i = 0; i < rows; i++) {
      final row = offset + i * stride;
      for (var j = 0; j < cols; j++) {
        storage[row + j] = value;
      }
    }
  }

  /// Sets this block to the identity. Only meaningful for a square view.
  void setIdentity() {
    fill(0);
    for (var i = 0; i < rows; i++) {
      set(i, i, 1);
    }
  }
}
