let fallbackCounter = 0

/** Create a browser-local ID without requiring crypto.randomUUID or HTTPS. */
export function createClientId(): string {
    const cryptoApi = globalThis.crypto

    if (typeof cryptoApi?.randomUUID === "function") {
        return cryptoApi.randomUUID()
    }

    if (typeof cryptoApi?.getRandomValues === "function") {
        const bytes = cryptoApi.getRandomValues(new Uint8Array(16))
        // Use the UUID v4 version and variant bits so the fallback has a familiar shape.
        bytes[6] = (bytes[6] & 0x0f) | 0x40
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        const hex = Array.from(bytes, (byte) =>
            byte.toString(16).padStart(2, "0"),
        ).join("")
        return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
    }

    // This final fallback is sufficient for temporary UI/history keys.
    fallbackCounter += 1
    return `local-${Date.now().toString(36)}-${fallbackCounter.toString(36)}-${Math.random().toString(36).slice(2)}`
}
