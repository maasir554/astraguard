import { test } from "node:test";
import assert from "node:assert/strict";
import {
  validatePolygon,
  parseCoordinates,
  hectares,
  SAMPLE,
  polygon,
} from "./geometry.ts";
test("valid region has positive geodesic area and a closed GeoJSON ring", () => {
  assert.equal(validatePolygon(SAMPLE), null);
  assert.ok(hectares(SAMPLE) > 0);
  const ring = polygon(SAMPLE).geometry.coordinates[0];
  assert.deepEqual(ring[0], ring.at(-1));
  assert.equal(ring.length, SAMPLE.length + 1);
});
test("rejects crossed boundaries", () =>
  assert.match(
    validatePolygon([
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
    ])!,
    /cross/,
  ));
test("rejects degenerate, duplicate, out-of-bounds and incomplete geometry", () => {
  assert.ok(
    validatePolygon([
      [0, 0],
      [1, 0],
      [2, 0],
    ]),
  );
  assert.ok(
    validatePolygon([
      [0, 0],
      [1, 1],
      [0, 0],
    ]),
  );
  assert.ok(
    validatePolygon([
      [181, 0],
      [1, 1],
      [0, 1],
    ]),
  );
  assert.ok(validatePolygon([[0, 0]]));
});
test("coordinate input validates shape and parses longitude first", () => {
  assert.deepEqual(parseCoordinates("77,23\n77.1,23\n77.1,23.1"), [
    [77, 23],
    [77.1, 23],
    [77.1, 23.1],
  ]);
  assert.throws(() => parseCoordinates("1,2,3\n4,5,6\n7,8,9"));
  assert.throws(() => parseCoordinates("abc,23\n77,24\n77,25"));
});
