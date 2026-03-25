// src/app/components/SettingsPanel.tsx

export default function SettingsPanel() {
  return (
    <div className="space-y-6">
      {/* LLM Model Selection */}
      <div>
        <label htmlFor="llm-model" className="block text-sm font-medium text-gray-700 mb-2">
          LLM Model
        </label>
        <select
          id="llm-model"
          name="llm-model"
          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
          defaultValue="gpt-3.5-turbo"
        >
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="gpt-4">GPT-4</option>
          {/* Add more models as needed */}
        </select>
      </div>

      {/* System Prompt Customization */}
      <div>
        <label htmlFor="system-prompt" className="block text-sm font-medium text-gray-700 mb-2">
          System Prompt
        </label>
        <textarea
          id="system-prompt"
          name="system-prompt"
          rows={5}
          className="shadow-sm focus:ring-blue-500 focus:border-blue-500 mt-1 block w-full sm:text-sm border border-gray-300 rounded-md p-2"
          placeholder="You are a helpful AI assistant."
        ></textarea>
      </div>

      {/* API Parameter Adjustment: Temperature */}
      <div>
        <label htmlFor="temperature" className="block text-sm font-medium text-gray-700 mb-2">
          Temperature
        </label>
        <input
          type="range"
          id="temperature"
          name="temperature"
          min="0" max="2" step="0.1"
          defaultValue="0.7"
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <span className="text-sm text-gray-500">0.7</span> {/* Display current value */}
      </div>

      {/* API Parameter Adjustment: Max Tokens */}
      <div>
        <label htmlFor="max-tokens" className="block text-sm font-medium text-gray-700 mb-2">
          Max Tokens
        </label>
        <input
          type="number"
          id="max-tokens"
          name="max-tokens"
          min="1" max="4000"
          defaultValue="1000"
          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
        />
      </div>
    </div>
  );
}
