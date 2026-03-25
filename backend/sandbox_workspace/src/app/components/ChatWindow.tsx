// src/app/components/ChatWindow.tsx

export default function ChatWindow() {
  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-md p-4">
      {/* Message display area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Example messages (will be replaced by dynamic messages) */}
        <div className="flex justify-start">
          <div className="bg-gray-200 p-3 rounded-lg max-w-xs">
            <p className="text-gray-800">Hello! How can I help you today?</p>
          </div>
        </div>
        <div className="flex justify-end">
          <div className="bg-blue-500 text-white p-3 rounded-lg max-w-xs">
            <p>Hi there! I'd like to ask a question.</p>
          </div>
        </div>
      </div>
      {/* Input area will be handled in page.tsx for now or a separate component */}
    </div>
  );
}
