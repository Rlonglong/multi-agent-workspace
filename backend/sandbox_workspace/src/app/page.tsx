import ChatWindow from './components/ChatWindow';
import SettingsPanel from './components/SettingsPanel';

export default function Home() {
  return (
    <div className="flex h-screen bg-gray-100">
      {/* Settings Panel */}
      <aside className="w-80 bg-white p-4 shadow-md">
        <h2 className="text-2xl font-semibold mb-6 text-gray-800">Settings</h2>
        <SettingsPanel />
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Chat Window */}
        <div className="flex-1 overflow-hidden">
          <ChatWindow />
        </div>

        {/* Message Input - This will eventually be part of ChatWindow or a separate component */}
        <div className="bg-gray-200 p-4">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              placeholder="Type your message..."
              className="flex-1 border rounded-lg py-2 px-4 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg">
              Send
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
