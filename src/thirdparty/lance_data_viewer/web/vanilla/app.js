class LanceViewer {
  constructor() {
    this.currentDataset = null;
    this.currentPage = 0;
    this.pageSize = 50;
    this.totalRows = 0;
    this.selectedColumns = [];
    this.allColumns = [];
    this.apiBase = window.location.origin;

    this.initializeElements();
    this.setupEventListeners();
    this.checkHealth();
    this.loadDatasets();
  }

  initializeElements() {
    this.elements = {
      toggleSidebar: document.getElementById("toggleSidebar"),
      mainContent: document.querySelector(".main-content"),
      healthStatus: document.getElementById("healthStatus"),
      datasetList: document.getElementById("datasetList"),
      datasetHeader: document.getElementById("datasetHeader"),
      columnSection: document.getElementById("columnSection"),
      schemaSection: document.getElementById("schemaSection"),
      schemaDisplay: document.getElementById("schemaDisplay"),
      dataSection: document.getElementById("dataSection"),
      dataTable: document.getElementById("dataTable"),
      tableHead: document.getElementById("tableHead"),
      tableBody: document.getElementById("tableBody"),
      dataLoading: document.getElementById("dataLoading"),
      dataError: document.getElementById("dataError"),
      columnList: document.getElementById("columnList"),
      prevPage: document.getElementById("prevPage"),
      nextPage: document.getElementById("nextPage"),
      pageInfo: document.getElementById("pageInfo"),
      pageSize: document.getElementById("pageSize"),
      selectAllCols: document.getElementById("selectAllCols"),
      selectNoneCols: document.getElementById("selectNoneCols"),
      applyColumns: document.getElementById("applyColumns"),
      tooltip: document.getElementById("tooltip"),
      toggleWordWrap: document.getElementById("toggleWordWrap"),
      toggleTruncate: document.getElementById("toggleTruncate"),
    };
  }

  setupEventListeners() {
    if (this.elements.toggleSidebar) {
      this.elements.toggleSidebar.addEventListener("click", () => {
        this.elements.mainContent.classList.toggle("sidebar-collapsed");
      });
    }
    this.elements.prevPage.addEventListener("click", () => this.previousPage());
    this.elements.nextPage.addEventListener("click", () => this.nextPage());
    this.elements.pageSize.addEventListener("change", (e) => {
      this.pageSize = parseInt(e.target.value);
      this.currentPage = 0;
      this.loadData();
    });

    this.elements.selectAllCols.addEventListener("click", () => this.selectAllColumns());
    this.elements.selectNoneCols.addEventListener("click", () => this.selectNoColumns());
    this.elements.applyColumns.addEventListener("click", () => this.applyColumnSelection());

    document.addEventListener("mousemove", (e) => this.updateTooltipPosition(e));

    this.elements.toggleWordWrap.addEventListener("change", (e) => {
      if (e.target.checked) {
        this.elements.dataTable.classList.add("wrap-text");
        this.elements.toggleTruncate.checked = false;
        this.elements.dataTable.classList.remove("truncate-text");
      } else {
        this.elements.dataTable.classList.remove("wrap-text");
      }
    });

    if (this.elements.toggleWordWrap.checked) {
      this.elements.dataTable.classList.add("wrap-text");
    }

    this.elements.toggleTruncate.addEventListener("change", (e) => {
      if (e.target.checked) {
        this.elements.dataTable.classList.add("truncate-text");
        this.elements.dataTable.style.tableLayout = "fixed";
        this.elements.toggleWordWrap.checked = false;
        this.elements.dataTable.classList.remove("wrap-text");
      } else {
        this.elements.dataTable.classList.remove("truncate-text");
        if (!this.elements.toggleWordWrap.checked) {
          this.elements.dataTable.style.tableLayout = "auto";
        }
      }
    });

    this.elements.tableBody.addEventListener("mouseover", (e) => {
      const td = e.target.closest("td");
      if (!td) return;

      if (
        this.elements.dataTable.classList.contains("truncate-text") &&
        !td.classList.contains("vector-cell") &&
        !td.querySelector(".complex-object-wrapper")
      ) {
        td.title = td.textContent;
      } else {
        td.removeAttribute("title");
      }
    });
  }

  async checkHealth() {
    try {
      const response = await fetch(`${this.apiBase}/healthz`);
      const data = await response.json();
      if (data.ok) {
        const lanceVersion = data.lancedb_version || "unknown";
        const pyarrowVersion = data.pyarrow_version || "unknown";
        this.elements.healthStatus.innerHTML = `
                    <div class="version-info">
                        <div class="app-version">Lance Data Viewer v${data.app_version}</div>
                        <div class="lance-version">LanceDB ${lanceVersion} • PyArrow ${pyarrowVersion}</div>
                    </div>
                `;
        this.elements.healthStatus.className = "health-status healthy";
      } else {
        throw new Error("Health check failed");
      }
    } catch (error) {
      this.elements.healthStatus.textContent = "Connection Error";
      this.elements.healthStatus.className = "health-status error";
    }
  }

  async loadDatasets() {
    try {
      const response = await fetch(`${this.apiBase}/datasets`);
      const data = await response.json();

      this.elements.datasetList.innerHTML = "";

      if (data.datasets.length === 0) {
        this.elements.datasetList.innerHTML = '<div class="loading">No datasets found</div>';
        return;
      }

      data.datasets.forEach((dataset) => {
        const item = document.createElement("div");
        item.className = "dataset-item";
        item.textContent = dataset;
        item.addEventListener("click", () => this.selectDataset(dataset));
        this.elements.datasetList.appendChild(item);
      });
    } catch (error) {
      this.elements.datasetList.innerHTML = '<div class="error">Failed to load datasets</div>';
    }
  }

  async selectDataset(datasetName) {
    document.querySelectorAll(".dataset-item").forEach((item) => {
      item.classList.remove("active");
    });

    event.target.classList.add("active");

    this.currentDataset = datasetName;
    this.currentPage = 0;
    this.elements.datasetHeader.style.display = "flex";

    await this.loadSchema();
    await this.loadColumns();
    await this.loadData();
  }

  async loadSchema() {
    try {
      const response = await fetch(`${this.apiBase}/datasets/${this.currentDataset}/schema`);
      const schema = await response.json();

      this.schemaFields = schema.fields;
    } catch (error) {
      console.error("Failed to load schema", error);
    }
  }
  async loadColumns() {
    try {
      const response = await fetch(`${this.apiBase}/datasets/${this.currentDataset}/columns`);
      const data = await response.json();

      this.allColumns = data.columns;
      this.selectedColumns = data.columns.map((col) => col.name);

      const list = this.elements.columnList;
      list.innerHTML = "";

      data.columns.forEach((column) => {
        const schemaField = this.schemaFields?.find((f) => f.name === column.name);
        const typeStr = schemaField ? schemaField.type : column.is_vector ? "vector" : "unknown";

        list.appendChild(this._makeColItem(column.name, typeStr, column.is_vector));
      });

      this._setupColumnDrag();

      list.style.display = "flex";
      list.parentElement.querySelector(".column-controls").style.display = "flex";
      this.elements.columnSection.style.display = "flex";
    } catch (error) {
      this.showError("Failed to load columns");
    }
  }
  _makeColItem(name, typeStr, isVector) {
    const id = `col_${name.replace(/[^a-z0-9]/gi, "_")}`;
    const item = document.createElement("div");
    item.className = "col-item";
    item.draggable = true;
    item.dataset.col = name;

    const handle = document.createElement("span");
    handle.className = "drag-handle";
    handle.textContent = "⠿";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = id;
    cb.checked = true;

    const label = document.createElement("label");
    label.htmlFor = id;

    label.innerHTML = `
        <span class="col-name">${name}</span>
        <span class="col-type">${typeStr}</span>
    `;
    label.title = `${name} (${typeStr})`;

    item.appendChild(handle);
    item.appendChild(cb);
    item.appendChild(label);
    return item;
  }
  _setupColumnDrag() {
    const list = this.elements.columnList;
    let dragging = null;

    list.addEventListener("dragstart", (e) => {
      dragging = e.target.closest(".col-item");
      if (!dragging) return;
      setTimeout(() => dragging.classList.add("dragging"), 0);
    });

    list.addEventListener("dragend", () => {
      if (dragging) dragging.classList.remove("dragging");
      list.querySelectorAll(".col-item").forEach((el) => el.classList.remove("drag-over"));
      dragging = null;
    });

    list.addEventListener("dragover", (e) => {
      e.preventDefault();
      const target = e.target.closest(".col-item");
      list.querySelectorAll(".col-item").forEach((el) => el.classList.remove("drag-over"));
      if (!target || target === dragging) return;
      const rect = target.getBoundingClientRect();
      if (e.clientY < rect.top + rect.height / 2) {
        list.insertBefore(dragging, target);
      } else {
        list.insertBefore(dragging, target.nextSibling);
      }
    });
  }

  selectAllColumns() {
    this.elements.columnList.querySelectorAll('input[type="checkbox"]').forEach((cb) => (cb.checked = true));
  }

  selectNoColumns() {
    this.elements.columnList.querySelectorAll('input[type="checkbox"]').forEach((cb) => (cb.checked = false));
  }

  applyColumnSelection() {
    this.selectedColumns = Array.from(this.elements.columnList.querySelectorAll(".col-item"))
      .filter((item) => item.querySelector('input[type="checkbox"]').checked)
      .map((item) => item.dataset.col);
    this.currentPage = 0;
    this.loadData();
  }

  async loadData() {
    if (!this.currentDataset) return;

    this.showLoading();

    try {
      const params = new URLSearchParams({
        limit: this.pageSize.toString(),
        offset: (this.currentPage * this.pageSize).toString(),
      });

      if (this.selectedColumns.length > 0 && this.selectedColumns.length < this.allColumns.length) {
        params.append("columns", this.selectedColumns.join(","));
      }

      const response = await fetch(`${this.apiBase}/datasets/${this.currentDataset}/rows?${params}`);
      const data = await response.json();

      this.totalRows = data.total;
      this.renderTable(data.rows);
      this.updatePagination();
      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      this.showError("Failed to load data");
    }
  }
  _calculateInitialWidth(columnName, rows) {
    let maxChars = columnName.length;
    let isVector = false;
    let isComplex = false;

    const rowsToCheck = Math.min(rows.length, 5);
    for (let i = 0; i < rowsToCheck; i++) {
      const val = rows[i][columnName];
      if (val !== null && val !== undefined) {
        if (typeof val === "object") {
          if (val.type === "vector") isVector = true;
          else isComplex = true;
        } else {
          const strLen = String(val).length;
          maxChars = Math.max(maxChars, Math.min(strLen, 80));
        }
      }
    }

    if (isVector) return 210;
    if (isComplex) {
      return Math.max(250, columnName.length * 10);
    }
    return Math.max(80, Math.min(maxChars * 8 + 35, 500));
  }
  _setupColumnResizer(resizer, th) {
    let x = 0;
    let w = 0;

    const mouseDownHandler = (e) => {
      x = e.clientX;
      const styles = window.getComputedStyle(th);
      w = parseInt(styles.width, 10);

      document.addEventListener("mousemove", mouseMoveHandler);
      document.addEventListener("mouseup", mouseUpHandler);

      resizer.classList.add("resizing");
      document.body.style.cursor = "col-resize";
    };

    const mouseMoveHandler = (e) => {
      const dx = e.clientX - x;
      const newWidth = Math.max(50, w + dx);
      th.style.width = `${newWidth}px`;
      th.style.minWidth = `${newWidth}px`;
    };

    const mouseUpHandler = () => {
      document.removeEventListener("mousemove", mouseMoveHandler);
      document.removeEventListener("mouseup", mouseUpHandler);

      resizer.classList.remove("resizing");
      document.body.style.cursor = "";
    };

    resizer.addEventListener("mousedown", mouseDownHandler);
  }
  renderTable(rows) {
    if (rows.length === 0) {
      this.elements.tableBody.innerHTML = '<tr><td colspan="100%">No data found</td></tr>';
      return;
    }

    const rowKeys = Object.keys(rows[0]);
    const columns = this.selectedColumns.length > 0 ? this.selectedColumns.filter((c) => rowKeys.includes(c)) : rowKeys;

    this.elements.tableHead.innerHTML = "";
    const headerRow = document.createElement("tr");

    columns.forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column;

      const initWidth = this._calculateInitialWidth(column, rows);

      th.style.width = `${initWidth}px`;
      th.style.minWidth = `${initWidth}px`;

      const resizer = document.createElement("div");
      resizer.className = "resizer";
      th.appendChild(resizer);

      this._setupColumnResizer(resizer, th);
      headerRow.appendChild(th);
    });
    this.elements.tableHead.appendChild(headerRow);

    this.elements.tableBody.innerHTML = "";
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((column) => {
        const td = document.createElement("td");
        const value = row[column];

        if (value && typeof value === "object") {
          if (value.type === "vector") {
            this.renderVectorCell(td, value, column);
          } else {
            this.renderComplexObject(td, value, column);
          }
        } else {
          td.textContent = value === null ? "null" : String(value);
        }

        tr.appendChild(td);
      });
      this.elements.tableBody.appendChild(tr);
    });

    this.elements.dataSection.style.display = "flex";
  }
  renderVectorCell(cell, vectorData, columnName) {
    cell.className = "vector-cell";

    if (vectorData.error) {
      cell.className = "vector-cell error";
      cell.textContent = `Vector Error: ${vectorData.error}`;
      return;
    }

    const container = document.createElement("div");
    container.className = "vector-preview";

    const info = document.createElement("div");
    info.className = "vector-info";

    if (vectorData.model === "likely_clip") {
      info.innerHTML = `
                <span class="vector-model">CLIP</span>
                <span class="vector-dim">dim: ${vectorData.dim}</span>
                <span class="vector-norm">norm: ${vectorData.norm.toFixed(3)}</span>
            `;
      if (vectorData.stats && vectorData.stats.normalized) {
        info.classList.add("normalized");
      }
    } else {
      info.textContent = `dim: ${vectorData.dim}, norm: ${vectorData.norm.toFixed(3)}`;
    }

    const canvas = document.createElement("canvas");
    canvas.className = "vector-sparkline";
    canvas.width = 180;
    canvas.height = 20;

    const ctx = canvas.getContext("2d");
    if (vectorData.preview && vectorData.preview.length > 0) {
      this.drawSparkline(ctx, vectorData.preview, canvas.width, canvas.height);
    }

    canvas.addEventListener("mouseenter", (e) => {
      this.showTooltip(e, vectorData, columnName);
    });

    canvas.addEventListener("mouseleave", () => {
      this.hideTooltip();
    });

    container.appendChild(info);
    container.appendChild(canvas);
    cell.appendChild(container);
  }

  drawSparkline(ctx, values, width, height) {
    const padding = 2;
    const drawWidth = width - 2 * padding;
    const drawHeight = height - 2 * padding;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = "#4dabf7";
    ctx.lineWidth = 2;
    ctx.beginPath();

    values.forEach((value, index) => {
      const x = padding + (index / (values.length - 1)) * drawWidth;
      const y = padding + (1 - (value - min) / range) * drawHeight;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();
  }
  showTooltip(event, vectorData, columnName) {
    const tooltip = this.elements.tooltip;
    const content = tooltip.querySelector(".tooltip-content");

    let tooltipHtml = `<strong>${columnName}</strong><br>`;

    if (vectorData.model === "likely_clip") {
      tooltipHtml += `
                <span class="model-badge">CLIP Embedding</span><br>
                ${vectorData.description}<br><br>
                Dimension: ${vectorData.dim}<br>
                Norm: ${vectorData.norm.toFixed(4)} ${vectorData.stats.normalized ? "(normalized ✓)" : ""}<br>
                Range: ${vectorData.min.toFixed(4)} to ${vectorData.max.toFixed(4)}<br>
                Mean: ${vectorData.mean.toFixed(4)}<br>
                Sparsity: ${(vectorData.stats.sparsity * 100).toFixed(1)}%<br>
                Positive ratio: ${(vectorData.stats.positive_ratio * 100).toFixed(1)}%<br><br>
                Preview: [${vectorData.preview
                  .slice(0, 8)
                  .map((v) => v.toFixed(3))
                  .join(", ")}...]
            `;
    } else {
      tooltipHtml += `
                Dimension: ${vectorData.dim}<br>
                Norm: ${vectorData.norm.toFixed(4)}<br>
                Min: ${vectorData.min.toFixed(4)}<br>
                Max: ${vectorData.max.toFixed(4)}<br>
                Preview: [${vectorData.preview
                  .slice(0, 8)
                  .map((v) => v.toFixed(2))
                  .join(", ")}...]
            `;
    }

    content.innerHTML = tooltipHtml;
    tooltip.style.display = "block";
    this.updateTooltipPosition(event);
  }

  hideTooltip() {
    this.elements.tooltip.style.display = "none";
  }

  updateTooltipPosition(event) {
    const tooltip = this.elements.tooltip;
    if (tooltip.style.display === "none") return;

    tooltip.style.left = event.pageX + 10 + "px";
    tooltip.style.top = event.pageY - 10 + "px";
  }

  updatePagination() {
    const totalPages = Math.ceil(this.totalRows / this.pageSize);
    const currentPageDisplay = this.currentPage + 1;

    this.elements.pageInfo.textContent = `Page ${currentPageDisplay} of ${totalPages} (${this.totalRows} total)`;
    this.elements.prevPage.disabled = this.currentPage === 0;
    this.elements.nextPage.disabled = this.currentPage >= totalPages - 1;
  }

  previousPage() {
    if (this.currentPage > 0) {
      this.currentPage--;
      this.loadData();
    }
  }

  nextPage() {
    const maxPage = Math.ceil(this.totalRows / this.pageSize) - 1;
    if (this.currentPage < maxPage) {
      this.currentPage++;
      this.loadData();
    }
  }

  showLoading() {
    this.elements.dataLoading.style.display = "block";
    this.elements.dataError.style.display = "none";
  }

  hideLoading() {
    this.elements.dataLoading.style.display = "none";
  }

  showError(message) {
    this.elements.dataError.textContent = message;
    this.elements.dataError.style.display = "block";
    this.elements.dataLoading.style.display = "none";
  }

  renderComplexObject(container, obj, name) {
    const wrapper = document.createElement("div");
    wrapper.className = "complex-object-wrapper";

    // NEW: Allow clicking the wrapper to expand it when in Truncate mode
    wrapper.addEventListener("click", (e) => {
      const isTruncated = this.elements.dataTable.classList.contains("truncate-text");
      // Only toggle if we are truncated AND we didn't just click the "Show more" button
      if (isTruncated && e.target.tagName !== "BUTTON") {
        wrapper.classList.toggle("expanded");
        e.stopPropagation();
      }
    });

    this._buildObjectDOM(wrapper, obj, name);
    container.appendChild(wrapper);
  }
  _buildObjectDOM(parent, obj, currentKey) {
    if (obj === null) {
      const span = document.createElement("span");
      span.className = "co-null";
      span.textContent = "null";
      parent.appendChild(span);
      return;
    }

    if (typeof obj !== "object") {
      const span = document.createElement("span");
      span.className = typeof obj === "string" ? "co-string" : "co-primitive";
      span.textContent = typeof obj === "string" ? `"${obj}"` : String(obj);
      parent.appendChild(span);
      return;
    }

    if (obj.type === "vector") {
      const vecWrap = document.createElement("div");
      this.renderVectorCell(vecWrap, obj, currentKey);
      parent.appendChild(vecWrap);
      return;
    }

    if (Array.isArray(obj)) {
      const list = document.createElement("div");
      list.className = "co-list";

      if (obj.length === 0) {
        const span = document.createElement("span");
        span.className = "co-empty";
        span.textContent = "[]";
        parent.appendChild(span);
        return;
      }

      const LIMIT = 5;
      const hasHidden = obj.length > LIMIT;
      let hiddenContainer = null;

      obj.forEach((item, index) => {
        const itemDiv = document.createElement("div");
        itemDiv.className = "co-list-item";

        const prefix = document.createElement("span");
        prefix.className = "co-prefix";
        prefix.textContent = `[${index}]: `;
        itemDiv.appendChild(prefix);

        if (item && typeof item === "object") {
          const childContainer = document.createElement("div");
          childContainer.className = "co-child-container";
          this._buildObjectDOM(childContainer, item, `${currentKey}[${index}]`);
          itemDiv.appendChild(childContainer);
        } else {
          this._buildObjectDOM(itemDiv, item, `${currentKey}[${index}]`);
        }

        if (hasHidden && index >= LIMIT) {
          if (!hiddenContainer) {
            hiddenContainer = document.createElement("div");
            hiddenContainer.style.display = "none";
            list.appendChild(hiddenContainer);
          }
          hiddenContainer.appendChild(itemDiv);
        } else {
          list.appendChild(itemDiv);
        }
      });

      if (hasHidden) {
        const toggleBtn = document.createElement("button");
        toggleBtn.className = "co-toggle-btn";
        toggleBtn.textContent = `Show ${obj.length - LIMIT} more...`;
        toggleBtn.onclick = (e) => {
          e.stopPropagation();
          if (hiddenContainer.style.display === "none") {
            hiddenContainer.style.display = "block";
            toggleBtn.textContent = "Show less";
          } else {
            hiddenContainer.style.display = "none";
            toggleBtn.textContent = `Show ${obj.length - LIMIT} more...`;
          }
        };
        list.appendChild(toggleBtn);
      }

      parent.appendChild(list);
      return;
    }

    const dict = document.createElement("div");
    dict.className = "co-dict";

    const keys = Object.keys(obj);
    if (keys.length === 0) {
      const span = document.createElement("span");
      span.className = "co-empty";
      span.textContent = "{}";
      parent.appendChild(span);
      return;
    }

    const LIMIT = 5;
    const hasHidden = keys.length > LIMIT;
    let hiddenContainer = null;

    keys.forEach((key, index) => {
      const row = document.createElement("div");
      row.className = "co-dict-row";

      const keySpan = document.createElement("strong");
      keySpan.className = "co-key";
      keySpan.textContent = `${key}: `;
      row.appendChild(keySpan);

      const val = obj[key];
      if (val && typeof val === "object") {
        const childContainer = document.createElement("div");
        childContainer.className = "co-child-container";
        this._buildObjectDOM(childContainer, val, key);
        row.appendChild(childContainer);
      } else {
        this._buildObjectDOM(row, val, key);
      }

      if (hasHidden && index >= LIMIT) {
        if (!hiddenContainer) {
          hiddenContainer = document.createElement("div");
          hiddenContainer.style.display = "none";
          dict.appendChild(hiddenContainer);
        }
        hiddenContainer.appendChild(row);
      } else {
        dict.appendChild(row);
      }
    });

    if (hasHidden) {
      const toggleBtn = document.createElement("button");
      toggleBtn.className = "co-toggle-btn";
      toggleBtn.textContent = `Show ${keys.length - LIMIT} more...`;
      toggleBtn.onclick = (e) => {
        e.stopPropagation();
        if (hiddenContainer.style.display === "none") {
          hiddenContainer.style.display = "block";
          toggleBtn.textContent = "Show less";
        } else {
          hiddenContainer.style.display = "none";
          toggleBtn.textContent = `Show ${keys.length - LIMIT} more...`;
        }
      };
      dict.appendChild(toggleBtn);
    }

    parent.appendChild(dict);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new LanceViewer();
});
