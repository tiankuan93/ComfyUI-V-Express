import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 定义一个函数来预览视频
function previewVideo(node, file, type) {
  // 清除 node 元素中的所有子元素
  try {
    var el = document.getElementById("VEpreviewVideo");
    el.remove();
  } catch (error) {
    console.log(error);
  }
  var element = document.createElement("div");
  element.id = "VEpreviewVideo";
  const previewNode = node;

  // 创建一个新的 video 元素
  let videoEl = document.createElement("video");

  // 设置 video 元素的属性
  videoEl.controls = true;
  videoEl.style.width = "100%";

  let params = {
    filename: file,
    type: type,
  };
  // 更新 video 元素的 src 属性
  videoEl.src = api.apiURL("/view?" + new URLSearchParams(params));

  // 重新加载并播放视频
  videoEl.load();
  videoEl.play();

  // 清除 div 元素中的所有子元素
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }

  // 将 video 元素添加到 div 元素中
  element.appendChild(videoEl);

  node.previewWidget = node.addDOMWidget("videopreview", "preview", element, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return element.value;
    },
    setValue(v) {
      element.value = v;
    },
  });

  var previewWidget = node.previewWidget;

  previewWidget.computeSize = function (width) {
    if (this.aspectRatio && !this.parentEl.hidden) {
      let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
      if (!(height > 0)) {
        height = 0;
      }
      this.computedHeight = height + 10;
      return [width, height];
    }
    return [width, -4]; //no loaded src, widget should not display
  };
}

app.registerExtension({
  name: "ComfyUI-V-Express.VideoPreviewer",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name == "VEPreview_Video") {
      nodeType.prototype.onExecuted = function (data) {
        previewVideo(this, data.video[0], data.video[1]);
      };
    }
  },
});
