const fs = require('fs');
const zlib = require('zlib');

function createPNG(width, height, getPixel) {
  // Grayscale 8-bit: 1 byte per pixel + 1 filter byte per row
  const rowSize = 1 + width;
  const rawData = Buffer.alloc(rowSize * height);

  for (let y = 0; y < height; y++) {
    const rowOffset = y * rowSize;
    rawData[rowOffset] = 0; // Filter type 0 (None)
    for (let x = 0; x < width; x++) {
      rawData[rowOffset + 1 + x] = getPixel(x, y);
    }
  }

  const compressed = zlib.deflateSync(rawData);

  // PNG Signature
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  // IHDR Chunk
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(width, 0);
  ihdrData.writeUInt32BE(height, 4);
  ihdrData.writeUInt8(8, 8);   // Bit depth: 8
  ihdrData.writeUInt8(0, 9);   // Color type: 0 (Grayscale)
  ihdrData.writeUInt8(0, 10);  // Compression
  ihdrData.writeUInt8(0, 11);  // Filter
  ihdrData.writeUInt8(0, 12);  // Interlace
  const ihdrChunk = makeChunk('IHDR', ihdrData);

  // IDAT Chunk
  const idatChunk = makeChunk('IDAT', compressed);

  // IEND Chunk
  const iendChunk = makeChunk('IEND', Buffer.alloc(0));

  return Buffer.concat([signature, ihdrChunk, idatChunk, iendChunk]);
}

function makeChunk(type, data) {
  const len = data.length;
  const buf = Buffer.alloc(12 + len);
  buf.writeUInt32BE(len, 0);
  buf.write(type, 4, 4, 'ascii');
  data.copy(buf, 8);

  // CRC32 over type and data
  const crc = crc32(buf.subarray(4, 8 + len));
  buf.writeUInt32BE(crc, 8 + len);
  return buf;
}

// Standard CRC32
function crc32(buf) {
  let table = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      if (c & 1) c = 0xedb88320 ^ (c >>> 1);
      else c = c >>> 1;
    }
    table[n] = c;
  }
  let crc = 0 ^ (-1);
  for (let i = 0; i < buf.length; i++) {
    crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
  }
  return (crc ^ (-1)) >>> 0;
}

// Generate Target Ground Truth
const W = 801, H = 1112;

function isInsideRotatedEllipse(x, y, cx, cy, rx, ry, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const dx = x - cx;
  const dy = y - cy;
  const tdx = cos * dx + sin * dy;
  const tdy = -sin * dx + cos * dy;
  return ((tdx * tdx) / (rx * rx) + (tdy * tdy) / (ry * ry)) <= 1.0;
}

const targetPNG = createPNG(W, H, (x, y) => {
  // Left breast
  if (isInsideRotatedEllipse(x, y, 268, 505, 52, 60, -0.1)) return 255;
  // Right breast
  if (isInsideRotatedEllipse(x, y, 438, 525, 54, 62, 0.1)) return 255;
  // Genitalia
  if (isInsideRotatedEllipse(x, y, 232, 955, 34, 40, 0.0)) return 255;
  return 0;
});

fs.writeFileSync('experimental-poc/samples/real_sample_gt_target.png', targetPNG);
console.log('Saved real_sample_gt_target.png (' + targetPNG.length + ' bytes)');
